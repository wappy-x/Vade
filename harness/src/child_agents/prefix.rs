//! The frozen parent prefix a child starts from: rules and tool schemas, no transcript.
use crate::agent::{Conversation, Limits};
use crate::json::{self, Value};
use std::io;

/// Frozen parent system/developer messages (including project rules), the
/// parent's latest project instructions update, and exact tool schemas. A
/// child's task and scope are appended as fresh user content.
/// No parent transcript, previous-response ID, or checkpoint handle is inherited.
#[derive(Clone)]
pub struct StablePrefix {
    options: Value,
    messages: Vec<Value>,
    pub(crate) advertised: Vec<String>,
    output_option: &'static str,
}

impl StablePrefix {
    pub fn from_parent(parent: &Conversation) -> io::Result<Self> {
        let Value::Object(fields) = parent.options() else {
            unreachable!("Conversation validates options")
        };
        // Fail closed on stateful options rather than reuse a parent's handle.
        // Unknown provider options are not copied: they can hide continuation
        // semantics. Extend this allowlist only after agreeing their contract.
        let allowed = [
            "model",
            "stream",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "temperature",
            "top_p",
            "seed",
            "n",
            "max_tokens",
            "max_completion_tokens",
            "prompt_cache_key",
            "cache",
            "thinking",
            "reasoning_effort",
            "stream_options",
        ];
        if fields
            .iter()
            .any(|(key, _)| !allowed.contains(&key.as_str()))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "child prefix has unsupported provider/state options",
            ));
        }
        let options = Value::Object(
            fields
                .iter()
                .filter(|(key, _)| {
                    !matches!(
                        key.as_str(),
                        "prompt_cache_key"
                            | "max_tokens"
                            | "max_completion_tokens"
                            | "stream"
                            | "stream_options"
                    )
                })
                .cloned()
                .collect(),
        );
        let mut messages: Vec<Value> = parent.messages()[..parent.prefix_len()]
            .iter()
            .map(|m| m.value().clone())
            .collect();
        // The frozen system section can be out of date. The parent's latest
        // project instructions update goes too; no other history is copied.
        messages.extend(crate::instructions::latest_update(parent).cloned());
        let advertised = match parent.options().get("tools") {
            Some(Value::Array(tools)) => tools
                .iter()
                .filter_map(|tool| {
                    tool.get("function")?
                        .get("name")?
                        .as_str()
                        .map(str::to_owned)
                })
                .collect(),
            _ => Vec::new(),
        };
        Ok(Self {
            options,
            messages,
            advertised,
            output_option: if parent.options().get("max_completion_tokens").is_some() {
                "max_completion_tokens"
            } else {
                "max_tokens"
            },
        })
    }

    pub(crate) fn conversation(
        &self,
        task: &super::Task,
        scope: &super::Scope,
        limits: Limits,
    ) -> io::Result<Conversation> {
        let Value::Object(mut fields) = self.options.clone() else {
            unreachable!()
        };
        // Children render lifecycle states, not token deltas. JSON responses use
        // the existing typed runtime and keep the exact system/tools prefix.
        fields.push(("stream".into(), Value::Bool(false)));
        fields.push((
            self.output_option.into(),
            Value::Number(task.quota.output_tokens_per_turn.to_string()),
        ));
        let mut messages = self.messages.clone();
        let envelope = Value::Object(vec![
            ("task".into(), Value::String(task.goal.clone())),
            ("allowed_tools".into(), Value::Array(scope.tools.iter().cloned().map(Value::String).collect())),
            ("write_paths".into(), Value::Array(scope.write_paths.iter().cloned().map(Value::String).collect())),
            ("quota".into(), Value::Object(vec![
                ("model_turns".into(), Value::Number(task.quota.model_turns.to_string())),
                ("tool_calls".into(), Value::Number(task.quota.tool_calls.to_string())),
                ("tokens_including_input_and_reserved_output".into(), Value::Number(task.quota.tokens.to_string())),
                ("output_tokens_per_turn".into(), Value::Number(task.quota.output_tokens_per_turn.to_string())),
                ("wall_time_ms".into(), Value::Number(task.quota.wall_time.as_millis().to_string())),
            ])),
            ("child_spawning".into(), Value::Bool(false)),
            ("integration".into(), Value::String("Return findings to the parent. Only the parent integrates results. Tool scopes are enforced by the host. workspace_shell, only when explicitly in allowed_tools, leases workspace writes exclusively; file tools still require exact write_paths. Parent approval policy applies to every side effect.".into())),
        ]);
        messages.push(Value::Object(vec![
            ("role".into(), Value::String("user".into())),
            ("content".into(), Value::String(json::stringify(&envelope)?)),
        ]));
        fields.push(("messages".into(), Value::Array(messages)));
        Conversation::from_request(Value::Object(fields), limits)
    }
}
