//! JSON-lines CPU tokenizer probe. Usage: tokenize TOKENIZER_JSON
//! Each input: {"text":"..."}, {"ids":[1,2],"skip_special":false}, or
//! {"messages":[{"role":"user","content":"..."}],"enable_thinking":false}.
use engine_model::json::{self, Value, field};
use engine_tokenizer::{
    Tokenizer,
    chat::{self, Message, Options, Role, ToolCall},
};
use std::io::{BufRead, Read, Write};
fn run(t: &Tokenizer, v: &Value) -> Result<String, Box<dyn std::error::Error>> {
    let o = v.object()?;
    if let Some(text) = o.get("text") {
        let text = text.string()?;
        let ids = if o.get("ordinary") == Some(&Value::Bool(true)) {
            t.encode_ordinary(text)?
        } else {
            t.encode(text)?
        };
        return Ok(format!(
            "{{\"ids\":{:?},\"decoded\":{}}}",
            ids,
            chat::quote(&t.decode(&ids, false)?)
        ));
    }
    if let Some(ids) = o.get("ids") {
        let ids = ids
            .array()?
            .iter()
            .map(|v| Ok(u32::try_from(v.u64()?)?))
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        let skip = o.get("skip_special") == Some(&Value::Bool(true));
        let mut decoder = t.stream_decoder(skip);
        let mut chunks = Vec::new();
        for &id in &ids {
            chunks.push(chat::quote(&decoder.push(id)?))
        }
        chunks.push(chat::quote(&decoder.finish()));
        return Ok(format!(
            "{{\"decoded\":{},\"chunks\":[{}]}}",
            chat::quote(&t.decode(&ids, skip)?),
            chunks.join(",")
        ));
    }
    let mut messages = Vec::new();
    for m in field(o, "messages")?.array()? {
        let m = m.object()?;
        let role = match field(m, "role")?.string()? {
            "system" => Role::System,
            "developer" => Role::Developer,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => return Err("unsupported role".into()),
        };
        let content = match m.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Null) | None => String::new(),
            _ => return Err("content must be text".into()),
        };
        let mut msg = Message::new(role, content);
        msg.tool_call_id = m.get("tool_call_id").and_then(|v| v.string().ok()).map(str::to_owned);
        msg.tool_name = m.get("name").and_then(|v| v.string().ok()).map(str::to_owned);
        msg.reasoning_content = match if t.is_gemma4() { m.get("reasoning").filter(|v| matches!(v, Value::String(s) if !s.is_empty())).or_else(|| m.get("reasoning_content")) } else { m.get("reasoning_content") } {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        };
        if let Some(calls) = m.get("tool_calls") {
            for call in calls.array()? {
                let call = call.object()?;
                let id = call.get("id").and_then(|v| v.string().ok()).map(str::to_owned);
                let call = call.get("function").unwrap_or(&Value::Null);
                let call = call.object()?;
                let arg = field(call, "arguments")?;
                let arguments = match arg {
                    Value::String(s) => s.clone(),
                    _ => serialize(arg),
                };
                msg.tool_calls.push(ToolCall {
                    id,
                    name: field(call, "name")?.string()?.into(),
                    arguments,
                });
            }
        }
        messages.push(msg)
    }
    let options = Options {
        enable_thinking: o.get("enable_thinking").map_or(t.chat_options().enable_thinking, |v| v == &Value::Bool(true)),
        preserve_thinking: o.get("preserve_thinking") == Some(&Value::Bool(true)),
        add_generation_prompt: o.get("add_generation_prompt") != Some(&Value::Bool(false)),
        max_bytes: 4 * 1024 * 1024,
        tools: match o.get("tools_json") {
            Some(v) => v
                .array()?
                .iter()
                .map(|x| Ok(x.string()?.into()))
                .collect::<Result<_, engine_model::Error>>()?,
            None => Vec::new(),
        },
        ..Default::default()
    };
    let text = t.format_chat(&messages, &options)?;
    let ids = chat::encode(t, &messages, &options)?;
    Ok(format!(
        "{{\"text\":{},\"ids\":{:?}}}",
        chat::quote(&text),
        ids
    ))
}
fn serialize(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.clone(),
        Value::String(v) => chat::quote(v),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(serialize).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", chat::quote(k), serialize(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: tokenize TOKENIZER_JSON")?;
    let mut t = if std::path::Path::new(&path).is_dir() { Tokenizer::from_directory(path)? } else { Tokenizer::from_file(path)? };
    // Match HTTP intake bytes; retain token slack for evaluation filler fitting.
    let limits = engine_tokenizer::Limits { max_input_bytes: 4 * 1024 * 1024, ..Default::default() };
    t.set_limits(limits)?;
    let stdin = std::io::stdin();
    let mut lock = stdin.lock();
    let mut stdout = std::io::stdout().lock();
    loop {
        let mut line = String::new();
        let n = (&mut lock).take(limits.max_input_bytes as u64 + 1).read_line(&mut line)?;
        if n == 0 {
            break;
        }
        if n > limits.max_input_bytes {
            return Err("input line too long".into());
        }
        let result = json::parse(line.as_bytes(), json::Limits { bytes: limits.max_input_bytes,
            string_bytes: limits.max_input_bytes, nodes: limits.max_tokens.saturating_add(65536), ..Default::default() })
            .map_err(|e| e.to_string())
            .and_then(|v| run(&t, &v).map_err(|e| e.to_string()));
        match result {
            Ok(s) => writeln!(stdout, "{s}")?,
            Err(e) => writeln!(stdout, "{{\"error\":{}}}", chat::quote(&e))?,
        }
        stdout.flush()?
    }
    Ok(())
}
