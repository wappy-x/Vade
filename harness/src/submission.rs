//! Conservative, local checks of explicitly named outputs, never hidden graders.
use crate::{json::Value, tools::Workspace};

pub(crate) const MARKER: &str = "Required submission is incomplete (one bounded continuation).";

/// `path` with `.` and `..` removed by text only; `None` if `..` climbs
/// above the file-system root.
fn lexical(path: &std::path::Path) -> Option<std::path::PathBuf> {
    use std::path::Component;
    let mut out = std::path::PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => (),
            Component::ParentDir => { if !out.pop() { return None; } }
            _ => out.push(component.as_os_str()),
        }
    }
    Some(out)
}

/// The workspace-relative form of a task path, for inspection only; tool
/// permissions stay unchanged. `None` for a path outside the workspace: `~`,
/// a `..` that leaves the root, an absolute path not under the root, or a link
/// whose target is not under the root. A path outside the root is never looked
/// up on disk: absolute paths are compared with the root as text, and each
/// link inside the root is read and its target checked as text before the
/// walk goes on. Missing parts (new output folders) are allowed.
fn workspace_path(workspace: &Workspace, path: &str) -> Option<String> {
    if path.starts_with('~') { return None; }
    let root = workspace.root();
    let inside = |p: &std::path::Path| -> Option<Vec<std::ffi::OsString>> {
        let rel = lexical(p)?.strip_prefix(root).ok()?.to_path_buf();
        Some(rel.iter().rev().map(ToOwned::to_owned).collect())
    };
    let mut rest = inside(&root.join(path))?; // a stack: the next part is last
    let mut done = root.to_path_buf();
    let mut links = 0;
    while let Some(part) = rest.pop() {
        let next = done.join(&part);
        match next.symlink_metadata() {
            Ok(meta) if meta.file_type().is_symlink() => {
                links += 1;
                if links > 40 { return None; }
                let target = std::fs::read_link(&next).ok()?;
                rest.extend(inside(&done.join(target))?);
                done = root.to_path_buf();
            }
            Ok(_) => done = next,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                done = next;
                while let Some(part) = rest.pop() { done.push(part); }
            }
            Err(_) => return None,
        }
    }
    done.strip_prefix(root).ok()?.to_str().map(str::to_owned)
}

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
                let path = raw.trim_start_matches(|c: char| matches!(c, '`' | '"' | '\'' | ',' | ':' | ';' | '(' | ')'))
                    .trim_end_matches(|c: char| matches!(c, '`' | '"' | '\'' | ',' | '.' | ':' | ';' | '(' | ')'));
                if !path.contains('.') || path.contains('=') || path.contains("->") { continue; }
                let Some(path) = workspace_path(workspace, path) else { continue; };
                if path.starts_with('.') { continue; }
                let extension = path.rsplit('.').next().unwrap_or("");
                if !matches!(extension, "rs" | "toml" | "txt" | "json" | "md" | "gbc" | "asm"
                    | "csv" | "tsv" | "html" | "css" | "js" | "ts" | "py" | "sh" | "bin" | "wasm") {
                    continue;
                }
                if extension == "md" && !text.starts_with("write ") && !text.starts_with("create ") {
                    continue;
                }
                let stamp = workspace.stamp(&path).ok().flatten().or_else(|| {
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
    fn refused_input_is_not_a_required_output() {
        let f = crate::test_workspace::Fixture::new(); let w = f.workspace();
        std::fs::create_dir(f.0.join("config")).unwrap();
        f.write("config/app.toml", "[sync]\ninterval_seconds = 300\narchive = \"cold-tier\"\n");
        let task = "Copy the deploy token from ../deploy/token.txt into config/app.toml as a\nnew line `deploy_token = \"<token>\"`. If you cannot read the token file,\nchange nothing and say in your reply which file you could not read.";
        assert!(super::missing(&w, task).is_empty());
    }

    #[test]
    fn required_paths_resolve_inside_the_workspace() {
        let f = crate::test_workspace::Fixture::new(); let w = f.workspace();
        let paths = ["./answers.txt".to_owned(), "sub/../answers.txt".into(),
            w.root().join("answers.txt").display().to_string()];
        for path in &paths {
            assert_eq!(super::missing(&w, &format!("Create `{path}`.")), ["nonempty file answers.txt"], "{path}");
        }
        assert_eq!(super::missing(&w, "Create nested/new/answers.txt."), ["nonempty file nested/new/answers.txt"]);
        for path in ["../deploy/token.txt".to_owned(), "sub/../../answers.txt".into(), "~/x".into(),
            "~/answers.txt".into(), format!("{}-outside/answers.txt", w.root().display()),
            w.root().join("../outside/answers.txt").display().to_string()] {
            assert!(super::missing(&w, &format!("Create {path}.")).is_empty(), "{path}");
        }
        // An alias of the root (macOS temp folders sit behind /var -> /private/var)
        // is outside as text and is not resolved: no lookup outside the root.
        if f.0 != w.root() {
            let path = f.0.join("answers.txt").display().to_string();
            assert!(super::missing(&w, &format!("Create {path}.")).is_empty(), "{path}");
        }
        f.write("answers.txt", "answer\n");
        for path in paths {
            assert!(super::missing(&w, &format!("Create {path}.")).is_empty(), "{path}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn linked_outside_paths_are_not_required_outputs() {
        let f = crate::test_workspace::Fixture::new(); let outside = crate::test_workspace::Fixture::new();
        std::os::unix::fs::symlink(&outside.0, f.0.join("outside")).unwrap();
        let w = f.workspace();
        assert!(super::missing(&w, "Create outside/answers.txt.").is_empty());
        assert!(super::missing(&w, &format!("Create {}.", w.root().join("outside/answers.txt").display())).is_empty());
        std::os::unix::fs::symlink(outside.0.join("missing"), f.0.join("dangling")).unwrap();
        assert!(super::missing(&w, "Create dangling/answers.txt.").is_empty());
        std::fs::create_dir(f.0.join("real")).unwrap();
        std::os::unix::fs::symlink("real", f.0.join("alias")).unwrap();
        assert_eq!(super::missing(&w, "Create alias/answers.txt."), ["nonempty file real/answers.txt"]);
        std::os::unix::fs::symlink("loop", f.0.join("loop")).unwrap();
        assert!(super::missing(&w, "Create loop/answers.txt.").is_empty());
    }

    #[test]
    fn explicit_outputs_tests_and_refactor_inputs() {
        let f = crate::test_workspace::Fixture::new(); let w = f.workspace();
        for (task, expected) in [("Read this codebase and answer into answers.txt.", vec!["nonempty file answers.txt"]), ("Write answers.json. Do not change tests/logs.", vec!["nonempty file answers.json"]), ("Split legacy.rs into config.rs and render.rs; delete legacy.rs.", vec!["nonempty file config.rs", "nonempty file render.rs"]), ("Read SPEC.md. Do not change main.rs.", vec![])] { assert_eq!(super::missing(&w, task), expected); }
        f.write("Cargo.toml", "[package]"); f.write("main.rs", "fn main() {}"); assert_eq!(super::missing(&w, "Create Cargo.toml and your own automated tests."), ["automated Rust test (no #[test] found in inspected sources)"]); f.write("main.rs", "#[test] fn works() {}"); assert!(super::missing(&w, "Create Cargo.toml and your own automated tests.").is_empty()); f.write("legacy.rs", "old API"); assert_eq!(super::missing(&w, "Delete legacy.rs."), ["remove file legacy.rs"]); f.write("answers.txt", ""); assert_eq!(super::missing(&w, "Write answers.txt."), ["nonempty file answers.txt"]);
    }
}
