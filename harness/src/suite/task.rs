//! The task file (`task.txt`) and the scripted transcripts (`solve.script`,
//! `wrong.script`).
//!
//! task.txt: `key: value` lines. A key with an empty value takes the indented
//! lines that follow (two leading spaces removed) as its value. `step:` starts
//! a new step; `prompt:`, `cancel_at_call:`, `remove_before:` (workspace paths
//! deleted before the step runs) and `step_check:` (checks run right after the
//! step) belong to the last step. `check:` lines run in order after the last
//! step. `inputs:` names workspace files the task reads. `#` starts a comment line.
use super::{n, obj, s, write};
use super::serve::Reply;
use crate::json::{self, Value};

#[derive(Clone, Debug)]
pub struct Step { pub kind: String, pub prompt: String, pub cancel_at_call: Option<usize>, pub remove_before: Vec<String>, pub checks: Vec<String> }

#[derive(Clone, Debug, Default)]
pub struct Task {
    pub id: String,
    pub title: String,
    /// One word that groups the task on the card, such as `bug-fix`.
    pub category: String,
    pub mode: String,
    pub budget_seconds: u64,
    pub generate: Option<String>,
    pub request: Option<String>,
    pub needs: Vec<String>,
    /// Workspace paths the task reads; each must exist in the starting workspace.
    pub inputs: Vec<String>,
    pub steps: Vec<Step>,
    pub checks: Vec<String>,
    /// The quality rubric from `rubric.txt` next to `task.txt`.
    pub rubric: super::rubric::Rubric,
}

pub fn parse(text: &str) -> Result<Task, String> {
    let mut t = Task { mode: "agent".into(), budget_seconds: 900, ..Default::default() };
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        i += 1;
        if line.trim().is_empty() || line.starts_with('#') { continue; }
        let (key, value) = line.split_once(':').ok_or_else(|| format!("line {i}: expected key: value"))?;
        let mut value = value.trim().to_string();
        if value.is_empty() {
            let mut block = Vec::new();
            while i < lines.len() && (lines[i].starts_with("  ") || lines[i].trim().is_empty()) {
                block.push(lines[i].strip_prefix("  ").unwrap_or(""));
                i += 1;
            }
            while block.last().is_some_and(|l| l.is_empty()) { block.pop(); }
            value = block.join("\n");
        }
        fn step(t: &mut Task, i: usize) -> Result<&mut Step, String> { t.steps.last_mut().ok_or_else(|| format!("line {i}: a step key comes before any step")) }
        match key.trim() {
            "id" => t.id = value,
            "title" => t.title = value,
            "category" => t.category = value,
            "mode" => t.mode = value,
            "budget_seconds" => t.budget_seconds = value.parse().map_err(|_| format!("line {i}: bad budget"))?,
            "generate" => t.generate = Some(value),
            "request" => t.request = Some(value),
            "needs" => t.needs.push(value),
            "step" => t.steps.push(Step { kind: value, prompt: String::new(), cancel_at_call: None, remove_before: Vec::new(), checks: Vec::new() }),
            "prompt" => step(&mut t, i)?.prompt = value,
            "cancel_at_call" => step(&mut t, i)?.cancel_at_call = Some(value.parse().map_err(|_| format!("line {i}: bad call number"))?),
            "inputs" => t.inputs.extend(value.split_whitespace().map(str::to_string)),
            "remove_before" => step(&mut t, i)?.remove_before.extend(value.split_whitespace().map(str::to_string)),
            "step_check" => step(&mut t, i)?.checks.push(value),
            "check" => t.checks.push(value),
            other => return Err(format!("line {i}: unknown key {other}")),
        }
    }
    if t.id.is_empty() { return Err("task has no id".into()); }
    if t.category.is_empty() || t.category.contains(char::is_whitespace) { return Err(format!("{}: the task needs a one-word category", t.id)); }
    match t.mode.as_str() {
        "agent" => {
            if t.steps.is_empty() { return Err(format!("{}: agent task has no step", t.id)); }
            for s in &t.steps {
                if !matches!(s.kind.as_str(), "new" | "resume" | "cancel") { return Err(format!("{}: unknown step kind {}", t.id, s.kind)); }
                if s.prompt.is_empty() { return Err(format!("{}: step {} has no prompt", t.id, s.kind)); }
                if (s.kind == "cancel") != s.cancel_at_call.is_some() { return Err(format!("{}: cancel steps, and only they, need cancel_at_call", t.id)); }
            }
        }
        "direct" => if t.request.is_none() { return Err(format!("{}: direct task has no request", t.id)); },
        m => return Err(format!("{}: unknown mode {m}", t.id)),
    }
    if t.checks.is_empty() { return Err(format!("{}: task has no check", t.id)); }
    Ok(t)
}

/// Scripted transcript: each `>>> ` line starts one model reply.
///   `>>> call NAME` + JSON arguments (one or more lines)
///   `>>> patch` + raw patch text (sent as apply_patch {"patch": text})
///   `>>> answer` + the final text
///   `>>> json` + a raw chat-completions body, sent as is
pub fn parse_script(text: &str) -> Result<Vec<Reply>, String> {
    let mut out = Vec::new();
    let mut head: Option<String> = None;
    let mut body: Vec<&str> = Vec::new();
    let mut flush = |head: &Option<String>, body: &mut Vec<&str>| -> Result<(), String> {
        let Some(h) = head else { return Ok(()) };
        let n = out.len() + 1;
        let text = body.join("\n");
        body.clear();
        let reply = match h.split_whitespace().collect::<Vec<_>>().as_slice() {
            ["call", name] => {
                let args = json::parse(text.trim()).map_err(|e| format!("reply {n}: arguments for {name}: {e}"))?;
                tool_reply(n, name, &write(&args))
            }
            ["patch"] => tool_reply(n, "apply_patch", &write(&obj(vec![("patch", s(&format!("{}\n", text.trim_end())))]))),
            ["answer"] => answer_reply(n, text.trim()),
            ["json"] => {
                json::parse(text.trim()).map_err(|e| format!("reply {n}: raw body: {e}"))?;
                text.trim().to_string()
            }
            other => return Err(format!("reply {n}: unknown script line >>> {}", other.join(" "))),
        };
        out.push(Reply { status: 200, content_type: "application/json".into(), body: reply });
        Ok(())
    };
    for line in text.lines() {
        if let Some(h) = line.strip_prefix(">>> ") {
            flush(&head, &mut body)?;
            head = Some(h.trim().to_string());
        } else if head.is_some() {
            body.push(line);
        } else if !line.trim().is_empty() && !line.starts_with('#') {
            return Err("text before the first >>> line".into());
        }
    }
    flush(&head, &mut body)?;
    if out.is_empty() { return Err("the script has no reply".into()); }
    Ok(out)
}

fn tool_reply(n: usize, name: &str, arguments: &str) -> String {
    let call = obj(vec![
        ("id", s(&format!("call_{n}"))), ("type", s("function")),
        ("function", obj(vec![("name", s(name)), ("arguments", s(arguments))])),
    ]);
    let message = obj(vec![("role", s("assistant")), ("tool_calls", Value::Array(vec![call]))]);
    completion(n, "tool_calls", message)
}

pub fn answer_reply(n: usize, text: &str) -> String {
    completion(n, "stop", obj(vec![("role", s("assistant")), ("content", s(text))]))
}

fn completion(id: usize, finish: &str, message: Value) -> String {
    write(&obj(vec![
        ("id", s(&format!("replay-{id}"))), ("object", s("chat.completion")), ("model", s("replay")),
        ("choices", Value::Array(vec![obj(vec![("index", n(0)), ("finish_reason", s(finish)), ("message", message)])])),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn task_blocks_and_steps() {
        let t = parse("id: x\ncategory: bug-fix\nstep: new\nprompt:\n  line one\n\n  line two\ncheck: last-exit-zero\n").unwrap();
        assert_eq!(t.steps[0].prompt, "line one\n\nline two");
        assert_eq!(t.checks, vec!["last-exit-zero"]);
        assert_eq!(t.category, "bug-fix");
        assert!(parse("id: x\ncategory: c\nstep: cancel\nprompt: p\ncheck: c\n").is_err());
        assert!(parse("id: x\nstep: new\nprompt: p\ncheck: c\n").unwrap_err().contains("category"));
        assert!(parse("id: x\ncategory: c\nstep: new\nprompt: p\n").unwrap_err().contains("no check"));
    }
    #[test]
    fn script_replies() {
        let r = parse_script("# c\n>>> call workspace_list\n{\"path\":\".\"}\n>>> patch\n*** Begin Patch\n*** End Patch\n>>> answer\nDone.\n").unwrap();
        assert_eq!(r.len(), 3);
        let v = json::parse(&r[1].body).unwrap();
        let args = v.get("choices").unwrap().as_array().unwrap()[0].get("message").unwrap().get("tool_calls").unwrap().as_array().unwrap()[0]
            .get("function").unwrap().get("arguments").unwrap().as_str().unwrap().to_string();
        assert_eq!(json::parse(&args).unwrap().get("patch").unwrap().as_str().unwrap(), "*** Begin Patch\n*** End Patch\n");
        assert!(parse_script(">>> call x\n{bad\n").is_err());
        assert!(parse_script("text\n>>> answer\nx\n").is_err());
        assert!(parse_script("# only a comment\n").is_err());
    }
}
