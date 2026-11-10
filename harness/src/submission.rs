//! Conservative, local checks of explicitly named outputs, never hidden graders.
use crate::{json::Value, tools::Workspace};

pub(crate) const MARKER: &str = "Required submission is incomplete (one bounded continuation).";

pub(crate) fn prompt(
    c: &crate::agent::Conversation,
    tools: &dyn crate::agent::ToolExecutor,
) -> Option<(String, bool)> {
    let task_index = c.messages().iter().rposition(|m| {
        m.role() == "user" && m.value().get("name").is_none()
            && !m.value().get("content").and_then(Value::as_str)
                .is_some_and(|s| matches!(s.trim(), "Continue" | "Continue."))
    })?;
    let reminded = c.messages()[task_index..].iter().any(|m| {
        m.role() == "developer" && m.value().get("content").and_then(Value::as_str)
            .is_some_and(|s| s.starts_with(MARKER))
    });
    let content = c.messages()[task_index].value().get("content")?;
    let task = match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter_map(|v| v.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>().join("\n"),
        _ => return None,
    };
    let missing = tools.missing_deliverables(&task);
    (!missing.is_empty()).then(|| (format!("{MARKER}\nMissing required deliverables:\n- {}\nUse the existing tools and remaining task budget to supply these, then run the requested checks. Do not claim completion while they are missing.", missing.join("\n- ")), reminded))
}

pub(crate) fn missing(workspace: &Workspace, task: &str) -> Vec<String> {
    let mut missing = std::collections::BTreeSet::new();
    // Join wrapped lines first; sentence boundaries must not split extensions.
    let mut sentence = Vec::new();
    for word in task.split_whitespace().chain(std::iter::once(".")) {
        sentence.push(word);
        if !word.ends_with('.') { continue; }
        let text = sentence.join(" ").to_ascii_lowercase();
        let output = ["write ", "create ", "implement ", "build ", "split ", "change all", "into ", "delete "]
            .iter().any(|v| text.contains(v));
        if output && !["do not ", "don't ", "no changes"].iter().any(|v| text.starts_with(v)) {
            for (i, raw) in sentence.iter().enumerate() {
                if i > 0 && matches!(sentence[i - 1].to_ascii_lowercase().as_str(), "split" | "read") {
                    continue;
                }
                let path = raw.trim_matches(|c: char| matches!(c, '`' | '"' | '\'' | ',' | '.' | ':' | ';' | '(' | ')'));
                if !path.contains('.') || path.starts_with('.') || path.contains('=') || path.contains("->") { continue; }
                let extension = path.rsplit('.').next().unwrap_or("");
                if !matches!(extension, "rs" | "toml" | "txt" | "json" | "md" | "gbc" | "asm"
                    | "csv" | "tsv" | "html" | "css" | "js" | "ts" | "py" | "sh" | "bin" | "wasm") {
                    continue;
                }
                if extension == "md" && !text.starts_with("write ") && !text.starts_with("create ") {
                    continue;
                }
                let stamp = workspace.stamp(path).ok().flatten().or_else(|| {
                    (extension == "rs" && !path.contains('/')).then(|| workspace.stamp(&format!("src/{path}")).ok().flatten()).flatten()
                });
                let exists = stamp.as_ref().is_some_and(|s| !s.directory);
                let present = exists && stamp.is_some_and(|s| s.bytes > 0);
                let deletion = i > 0 && sentence[i - 1].eq_ignore_ascii_case("delete");
                if deletion && exists { missing.insert(format!("remove file {path}")); }
                if !deletion && !present { missing.insert(format!("nonempty file {path}")); }
            }
        }
        sentence.clear();
    }
    let lower = task.to_ascii_lowercase();
    if ["own automated tests", "add tests", "create tests", "write tests"].iter().any(|s| lower.contains(s)) {
        let mut pending = vec![(String::new(), 0)];
        let mut found = false;
        let mut complete = true;
        let mut inspected = 0;
        while let Some((dir, depth)) = pending.pop() {
            let entries = match workspace.list(&dir) { Ok(v) => v, Err(_) => { complete = false; continue; } };
            for name in entries {
                inspected += 1;
                if inspected > 256 { break; }
                if matches!(name.as_str(), "target" | "node_modules" | "tools") { continue; }
                let path = if dir.is_empty() { name } else { format!("{dir}/{name}") };
                if path.ends_with(".rs") {
                    match workspace.read(&path) {
                        Ok(s) if s.lines().any(|l| l.trim_start().starts_with("#[test]")) => { found = true; break; }
                        Err(_) => complete = false,
                        _ => (),
                    }
                }
                if workspace.stamp(&path).ok().flatten().is_some_and(|s| s.directory) {
                    if depth < 4 { pending.push((path, depth + 1)); } else { complete = false; }
                }
            }
            if found || inspected > 256 { break; }
        }
        if !found && complete && inspected <= 256 { missing.insert("automated Rust test (no #[test] found in inspected sources)".into()); }
    }
    missing.into_iter().collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn explicit_outputs_tests_and_refactor_inputs() {
        let f = crate::test_workspace::Fixture::new(); let w = f.workspace();
        for (task, expected) in [("Read this codebase and answer into answers.txt.", vec!["nonempty file answers.txt"]), ("Write answers.json. Do not change tests/logs.", vec!["nonempty file answers.json"]), ("Split legacy.rs into config.rs and render.rs; delete legacy.rs.", vec!["nonempty file config.rs", "nonempty file render.rs"]), ("Read SPEC.md. Do not change main.rs.", vec![])] { assert_eq!(super::missing(&w, task), expected); }
        f.write("Cargo.toml", "[package]"); f.write("main.rs", "fn main() {}"); assert_eq!(super::missing(&w, "Create Cargo.toml and your own automated tests."), ["automated Rust test (no #[test] found in inspected sources)"]); f.write("main.rs", "#[test] fn works() {}"); assert!(super::missing(&w, "Create Cargo.toml and your own automated tests.").is_empty()); f.write("legacy.rs", "old API"); assert_eq!(super::missing(&w, "Delete legacy.rs."), ["remove file legacy.rs"]); f.write("answers.txt", ""); assert_eq!(super::missing(&w, "Write answers.txt."), ["nonempty file answers.txt"]);
    }
}
