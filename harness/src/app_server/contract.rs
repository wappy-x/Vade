//! First-party validation of the declared app-server subset. See
//! docs/bridge-contract.md for the source revision and compatibility policy.
use super::{Event, Phase, invalid, valid_id};
use crate::json::{self, Value};
use std::io;

pub(super) const MAX_EVENTS: usize = 16_384;
const MAX_ITEMS: usize = 256;
const MAX_ITEM_BYTES: usize = 8 * 1024 * 1024;

pub(super) fn envelope(value: &Value) -> io::Result<()> {
    if !matches!(value, Value::Object(_)) {
        return Err(invalid("app-server message must be an object"));
    }
    if value
        .get("jsonrpc")
        .is_some_and(|v| v.as_str() != Some("2.0"))
    {
        return Err(invalid("unsupported app-server JSON-RPC version"));
    }
    if let Some(id) = value.get("id") {
        match id {
            Value::String(s) if valid_id(s) => (),
            Value::Number(n) if n.parse::<u64>().is_ok() => (),
            _ => return Err(invalid("invalid app-server message id")),
        }
    }
    if let Some(method) = value.get("method") {
        if !method.as_str().is_some_and(valid_id)
            || !matches!(value.get("params"), Some(Value::Object(_)))
            || value.get("result").is_some()
            || value.get("error").is_some()
        {
            return Err(invalid("malformed app-server method envelope"));
        }
    } else if value.get("id").is_none()
        || value.get("result").is_some() == value.get("error").is_some()
    {
        return Err(invalid("malformed app-server response envelope"));
    }
    Ok(())
}

pub(super) fn outbound(method: &str) -> io::Result<()> {
    if matches!(
        method,
        "initialize"
            | "thread/start"
            | "thread/resume"
            | "turn/start"
            | "turn/interrupt"
            | "turn/steer"
    ) {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "method outside Vade's delegated-agent contract",
        ))
    }
}

/// Version strings are diagnostic metadata, not capability negotiation. The
/// published initialize result does not provide a negotiated protocol version.
pub(super) fn initialize(result: &Value) -> io::Result<()> {
    if !result
        .get("userAgent")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty() && s.len() <= 4096 && !s.chars().any(char::is_control))
    {
        return Err(invalid("unsupported app-server initialize response"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Notification {
    Turn,
    Item,
    Delta,
    Error,
    Informational,
}

pub(super) fn notification(method: &str) -> io::Result<Notification> {
    Ok(match method {
        "turn/started" | "turn/completed" => Notification::Turn,
        "item/started" | "item/completed" => Notification::Item,
        "item/agentMessage/delta" => Notification::Delta,
        "error" => Notification::Error,
        "thread/started"
        | "thread/status/changed"
        | "thread/tokenUsage/updated"
        | "thread/name/updated"
        | "serverRequest/resolved"
        | "configWarning"
        | "warning"
        | "turn/diff/updated"
        | "turn/plan/updated"
        | "item/plan/delta"
        | "item/reasoning/summaryTextDelta"
        | "item/reasoning/summaryPartAdded"
        | "item/reasoning/textDelta"
        | "item/commandExecution/outputDelta"
        | "item/commandExecution/terminalInteraction"
        | "item/fileChange/outputDelta"
        | "item/mcpToolCall/progress"
        | "model/rerouted"
        | "model/safetyBuffering/updated"
        | "hook/started"
        | "hook/completed" => Notification::Informational,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "unknown app-server notification; contract update required",
            ));
        }
    })
}

struct Item {
    id: String,
    kind: String,
    completed: Option<Value>,
}

#[derive(Default)]
pub(super) struct Items {
    items: Vec<Item>,
    bytes: usize,
}

impl Items {
    fn entry(&mut self, id: &str, kind: &str) -> io::Result<&mut Item> {
        if !valid_id(id) {
            return Err(invalid("invalid app-server item id"));
        }
        let index = match self.items.iter().position(|item| item.id == id) {
            Some(index) => index,
            None => {
                if self.items.len() >= MAX_ITEMS {
                    return Err(invalid("app-server item count exceeds limit"));
                }
                self.items.push(Item {
                    id: id.into(),
                    kind: kind.into(),
                    completed: None,
                });
                self.items.len() - 1
            }
        };
        let item = &mut self.items[index];
        if item.kind != kind {
            return Err(invalid("app-server item changed type"));
        }
        Ok(item)
    }

    pub(super) fn delta(&mut self, params: &Value) -> io::Result<String> {
        let id = params
            .get("itemId")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("agent delta has no item id"))?;
        let item = self.entry(id, "agentMessage")?;
        if item.completed.is_some() {
            return Err(invalid("app-server delta arrived after item completion"));
        }
        params
            .get("delta")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| invalid("agent delta is not text"))
    }

    pub(super) fn update(&mut self, value: &Value, complete: bool) -> io::Result<Option<Event>> {
        let id = value
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("app-server item has no id"))?;
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("app-server item has no type"))?;
        let phase = match kind {
            "agentMessage" | "userMessage" | "reasoning" | "plan" | "enteredReviewMode"
            | "exitedReviewMode" => Phase::Generating,
            "commandExecution" | "fileChange" | "mcpToolCall" | "dynamicToolCall"
            | "collabToolCall" | "webSearch" | "imageView" | "imageGeneration"
            | "functionCallOutput" => Phase::ToolRunning,
            "contextCompaction" => Phase::Compacting,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "unknown app-server item type; contract update required",
                ));
            }
        };
        let item = self.entry(id, kind)?;
        if let Some(previous) = &item.completed {
            return if complete && previous == value {
                Ok(None)
            } else {
                Err(invalid("app-server changed an already completed item"))
            };
        }
        if !complete {
            return Ok(Some(Event::Phase(phase)));
        }
        if kind == "agentMessage" && value.get("text").and_then(Value::as_str).is_none() {
            return Err(invalid("completed agent item has no text"));
        }
        let size = json::stringify(value)?.len();
        if size > MAX_ITEM_BYTES.saturating_sub(self.bytes) {
            return Err(invalid("app-server completed item bytes exceed limit"));
        }
        self.bytes += size;
        self.entry(id, kind)?.completed = Some(value.clone());
        Ok(if kind == "agentMessage" {
            Some(Event::AgentMessageCompleted {
                item_id: id.into(),
                text: value.get("text").and_then(Value::as_str).unwrap().into(),
            })
        } else {
            Some(Event::Phase(Phase::Generating))
        })
    }

    pub(super) fn terminal_items(&mut self, turn: &Value) -> io::Result<()> {
        let values = turn
            .get("items")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("app-server turn has no item array"))?;
        for value in values {
            self.update(value, true)?;
        }
        Ok(())
    }

    pub(super) fn finish(&self) -> io::Result<String> {
        let mut text = String::new();
        for item in &self.items {
            let completed = item
                .completed
                .as_ref()
                .ok_or_else(|| invalid("app-server completed turn with unfinished items"))?;
            if item.kind == "agentMessage" {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(completed.get("text").and_then(Value::as_str).unwrap());
            }
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixture_completion_overrides_provisional_text() {
        let mut items = Items::default();
        let fixture = include_str!("../../tests/fixtures/codex-v2-items.jsonl");
        for line in fixture.lines() {
            let value = json::parse(line).unwrap();
            envelope(&value).unwrap();
            let params = value.get("params").unwrap();
            match value.get("method").and_then(Value::as_str).unwrap() {
                "item/agentMessage/delta" => {
                    assert_eq!(items.delta(params).unwrap(), "provisional draft");
                }
                "item/started" => {
                    items.update(params.get("item").unwrap(), false).unwrap();
                }
                "item/completed" => {
                    items.update(params.get("item").unwrap(), true).unwrap();
                }
                "turn/completed" => {
                    items.terminal_items(params.get("turn").unwrap()).unwrap();
                }
                _ => unreachable!(),
            }
        }
        assert_eq!(items.finish().unwrap(), "authoritative answer");
    }
    #[test]
    fn no_shell_or_generic_rpc_escape_hatch() {
        for method in [
            "thread/shellCommand",
            "command/exec",
            "fs/writeFile",
            "thread/inject_items",
            "account/login/start",
        ] {
            assert_eq!(
                outbound(method).unwrap_err().kind(),
                io::ErrorKind::Unsupported
            );
        }
    }
    #[test]
    fn envelope_versions_and_unknown_events_fail_explicitly() {
        for text in [
            r#"{"id":1,"result":{},"error":{}}"#,
            r#"{"id":null,"result":{}}"#,
            r#"{"method":"warning","params":[],"result":{}}"#,
            r#"{"jsonrpc":"3.0","id":1,"result":{}}"#,
        ] {
            assert!(envelope(&json::parse(text).unwrap()).is_err());
        }
        assert!(notification("turn/newTerminalStatus").is_err());
        assert!(initialize(&json::parse(r#"{"version":"future"}"#).unwrap()).is_err());
        for user_agent in ["mock-v1", "mock-v2"] {
            initialize(&super::super::object(vec![(
                "userAgent",
                super::super::string(user_agent),
            )]))
            .unwrap();
        }
    }
    #[test]
    fn incomplete_conflicting_and_late_items_are_rejected() {
        let mut items = Items::default();
        let delta = json::parse(r#"{"itemId":"i","delta":"draft"}"#).unwrap();
        items.delta(&delta).unwrap();
        assert!(items.finish().is_err());
        let complete = json::parse(r#"{"id":"i","type":"agentMessage","text":"final"}"#).unwrap();
        items.update(&complete, true).unwrap();
        assert!(items.update(&complete, true).unwrap().is_none());
        assert!(items.delta(&delta).is_err());
        assert!(
            items
                .update(
                    &json::parse(r#"{"id":"i","type":"agentMessage","text":"different"}"#).unwrap(),
                    true
                )
                .is_err()
        );
    }
}
