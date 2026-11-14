//! Text rendering of the official Llama 3.1/3.2 templates.
use super::{Message, Options, Output, Role};
use crate::{Result, err};
use engine_model::json;

pub(crate) fn validate(path: &std::path::Path) -> Result<bool> {
    let bytes = engine_model::safetensors::read_bounded(path, 2 * 1024 * 1024)?;
    let value = json::parse(&bytes, json::Limits { bytes: 2 * 1024 * 1024, ..Default::default() })?;
    let o = value.object()?;
    let template = json::field(o, "chat_template")?.string()?;
    let current_date = if template == include_str!("../../templates/llama3.jinja") { true }
        else if template == include_str!("../../templates/llama31.jinja") { false }
        else { return Err(err("unsupported Llama chat template")); };
    if json::field(o, "bos_token")?.string()? != "<|begin_of_text|>"
        || json::field(o, "eos_token")?.string()? != "<|eot_id|>"
        || json::field(o, "tokenizer_class")?.string()? != "PreTrainedTokenizerFast" {
        return Err(err("unsupported Llama chat controls"));
    }
    Ok(current_date)
}

pub(super) fn render(messages: &[Message], options: &Options, current_date: bool) -> Result<Output> {
    if messages.is_empty() || messages.len() > 4096 { return Err(err("Llama chat requires 1..4096 messages")); }
    if !options.tools.is_empty() || messages.iter().any(|m| m.role == Role::Tool || !m.tool_calls.is_empty()) {
        return Err(err("Llama tool chat is not yet supported"));
    }
    let mut out = Output { text: String::new(), limit: options.max_bytes, ordinary: Vec::new() };
    out.push("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: ")?;
    out.push(&if current_date { today()? } else { "26 Jul 2024".into() })?;
    out.push("\n\n")?;
    let system = messages.first().filter(|m| m.role == Role::System);
    if let Some(m) = system { out.data(m.content.trim())?; }
    out.push("<|eot_id|>")?;
    for m in &messages[usize::from(system.is_some())..] {
        out.push("<|start_header_id|>")?; out.push(m.role.name())?;
        out.push("<|end_header_id|>\n\n")?; out.data(m.content.trim())?; out.push("<|eot_id|>")?;
    }
    if options.add_generation_prompt { out.push("<|start_header_id|>assistant<|end_header_id|>\n\n")?; }
    Ok(out)
}

pub(crate) fn format(messages: &[Message], options: &Options, current_date: bool) -> Result<String> {
    Ok(render(messages, options, current_date)?.text)
}

fn today() -> Result<String> {
    let days = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| err("system clock predates Unix epoch"))?.as_secs() / 86400;
    let z = days as i64 + 719468;
    let era = z / 146097;
    let day = z - era * 146097;
    let year = (day - day / 1460 + day / 36524 - day / 146096) / 365;
    let doy = day - (365 * year + year / 4 - year / 100);
    let month = (5 * doy + 2) / 153;
    let d = doy - (153 * month + 2) / 5 + 1;
    let m = month + if month < 10 { 3 } else { -9 };
    let y = year + era * 400 + i64::from(m <= 2);
    let names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    Ok(format!("{d:02} {} {y}", names[(m - 1) as usize]))
}
