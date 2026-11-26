//! Plain text turns from the checked GLM template. Tools fail closed.
use super::{Message, Options, Output, ReasoningEffort, Role};
use crate::{Result, Tokenizer, err};
use std::{io::Read, path::Path};

pub(crate) fn has_controls(tokenizer: &Tokenizer) -> bool {
    ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>"].iter().all(|text|
        tokenizer.added.iter().any(|a| a.text == *text && a.special && !a.strip))
}
pub(crate) fn validate_directory(path: &Path) -> Result<()> {
    let mut bytes = Vec::new();
    std::fs::File::open(path.join("chat_template.jinja"))?.take(2 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes != include_bytes!("../../templates/glm.jinja") {
        return Err(err("unsupported GLM chat template configuration"));
    }
    // Some artifacts carry the template in both files. They must agree.
    let mut config = Vec::new();
    std::fs::File::open(path.join("tokenizer_config.json"))?.take(2 * 1024 * 1024 + 1).read_to_end(&mut config)?;
    let value = engine_model::json::parse(&config, engine_model::json::Limits { bytes: 2 * 1024 * 1024, ..Default::default() })?;
    if let Some(template) = value.object()?.get("chat_template")
        && template.string()?.as_bytes() != bytes { return Err(err("GLM chat templates disagree")); }
    Ok(())
}
pub(crate) fn format(messages: &[Message], options: &Options) -> Result<String> {
    Ok(render(messages, options)?.text)
}
pub(super) fn render(messages: &[Message], options: &Options) -> Result<Output> {
    if messages.is_empty() || messages.len() > 4096 { return Err(err("chat item count limit exceeded")); }
    if !options.tools.is_empty() || !options.generation_prefix.is_empty() {
        return Err(err("GLM plain chat does not support tools or a forced generation prefix"));
    }
    let mut out = Output { text: String::new(), limit: options.max_bytes, ordinary: Vec::new() };
    out.push("[gMASK]<sop><|system|>Reasoning Effort: ")?;
    // The template accepts low and high. All other values resolve to max.
    out.push(if options.reasoning_effort == ReasoningEffort::Low { "Low" } else { "Max" })?;
    for message in messages {
        if !message.tool_calls.is_empty() || message.tool_call_id.is_some() || message.tool_name.is_some() || message.reasoning_content.is_some() {
            return Err(err("GLM plain chat requires system or user text turns"));
        }
        out.push(match message.role {
            Role::System => "<|system|>", Role::User => "<|user|>",
            _ => return Err(err("GLM plain chat requires system or user text turns")),
        })?;
        out.data(&message.content)?;
    }
    // enable_thinking is not read by this checkpoint template.
    if options.add_generation_prompt { out.push("<|assistant|><think>")?; }
    Ok(out)
}
