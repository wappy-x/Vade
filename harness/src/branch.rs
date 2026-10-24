//! Transcript branches share a workspace, not file snapshots. Optional KV hints are Engine-owned.
//! Revision references are bounded local metadata checks, never content hashes.
use crate::agent::Conversation;
use crate::json::{self, Value};
use crate::tools::{Workspace, safe_parts};
use std::collections::BTreeSet;
use std::io;
const MAX_REFERENCES: usize = 128;

fn err(s: &str) -> io::Error {
    io::Error::other(s)
}
pub(crate) fn workspace_id(workspace: &Workspace) -> io::Result<String> {
    let stamp = workspace
        .stamp(".")?
        .ok_or_else(|| err("workspace unavailable"))?;
    Ok(stamp
        .identity
        .split(':')
        .take(2)
        .collect::<Vec<_>>()
        .join(":"))
}
fn path(candidate: &str) -> io::Result<String> {
    let parts = safe_parts(candidate)?;
    Ok(if parts.is_empty() {
        ".".into()
    } else {
        parts.join("/")
    })
}
#[derive(Clone, Debug)]
pub struct Checkpoint {
    workspace_id: String,
    files: Vec<(String, Option<String>)>,
    incomplete: bool,
    untracked_commands: bool,
}
impl Checkpoint {
    pub fn capture(workspace: &Workspace, c: &Conversation) -> io::Result<Self> {
        let mut paths = BTreeSet::new();
        let mut incomplete = false;
        let mut untracked_commands = false;
        for message in c.messages() {
            for call in message.tool_calls() {
                let Ok(args) = json::parse(&call.arguments) else {
                    incomplete = true;
                    continue;
                };
                let mut candidates = Vec::new();
                match call.name.as_str() {
                    "workspace_read" | "workspace_list" | "workspace_manifest"
                    | "workspace_symbols" | "workspace_read_lines" | "workspace_edit_lines" => {
                        if let Some(p) = args.get("path").and_then(Value::as_str) {
                            candidates.push(p);
                        }
                    }
                    "apply_patch" => {
                        if let Some(patch) = args.get("patch").and_then(Value::as_str) {
                            for line in patch.lines() {
                                for marker in [
                                    "*** Add File: ",
                                    "*** Update File: ",
                                    "*** Delete File: ",
                                    "*** Move to: ",
                                ] {
                                    if let Some(p) = line.strip_prefix(marker) {
                                        candidates.push(p);
                                    }
                                }
                            }
                        }
                    }
                    "workspace_shell" => untracked_commands = true,
                    "skill_read" => incomplete = true, // lazy body path is in its result; do not guess by parsing prose
                    "tool_output_read" => (),
                    _ => incomplete = true,
                }
                for candidate in candidates {
                    match path(candidate) {
                        Ok(path) if paths.contains(&path) || paths.len() < MAX_REFERENCES => {
                            paths.insert(path);
                        }
                        _ => incomplete = true,
                    }
                }
            }
        }
        let mut files = Vec::new();
        for path in paths {
            match workspace.stamp(&path) {
                Ok(stamp) => files.push((path, stamp.map(|s| s.identity))),
                Err(_) => incomplete = true, // unsafe/unavailable paths are not probed further
            }
        }
        Ok(Self {
            workspace_id: workspace_id(workspace)?,
            files,
            incomplete,
            untracked_commands,
        })
    }
    pub(crate) fn value(&self) -> Value {
        Value::Object(vec![
            (
                "workspace_id".into(),
                Value::String(self.workspace_id.clone()),
            ),
            (
                "files".into(),
                Value::Array(
                    self.files
                        .iter()
                        .map(|(path, revision)| {
                            Value::Object(vec![
                                ("path".into(), Value::String(path.clone())),
                                (
                                    "revision".into(),
                                    revision.clone().map_or(Value::Null, Value::String),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("incomplete".into(), Value::Bool(self.incomplete)),
            (
                "untracked_commands".into(),
                Value::Bool(self.untracked_commands),
            ),
        ])
    }
    pub(crate) fn from_value(value: &Value) -> io::Result<Self> {
        let workspace_id = value
            .get("workspace_id")
            .and_then(Value::as_str)
            .ok_or_else(|| err("checkpoint workspace missing"))?
            .to_string();
        if workspace_id.len() > 64
            || !workspace_id
                .bytes()
                .all(|b| b.is_ascii_digit() || b == b':')
        {
            return Err(err("invalid workspace identity"));
        }
        let raw = value
            .get("files")
            .and_then(Value::as_array)
            .ok_or_else(|| err("checkpoint file references missing"))?;
        if raw.len() > MAX_REFERENCES {
            return Err(err("too many checkpoint file references"));
        }
        let mut files = Vec::new();
        for item in raw {
            let name = item
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| err("invalid file reference"))?;
            if path(name)? != name {
                return Err(err("noncanonical file reference"));
            }
            let revision = match item.get("revision") {
                Some(Value::Null) => None,
                Some(Value::String(r))
                    if r.len() <= 200
                        && r.bytes()
                            .all(|b| b.is_ascii_digit() || b == b':' || b == b'-') =>
                {
                    Some(r.clone())
                }
                _ => return Err(err("invalid file revision")),
            };
            files.push((name.into(), revision));
        }
        if files.windows(2).any(|w| w[0].0 >= w[1].0) {
            return Err(err("duplicate or unordered file references"));
        }
        let boolean = |key| match value.get(key) {
            Some(Value::Bool(b)) => Ok(*b),
            _ => Err(err("invalid checkpoint flag")),
        };
        let checkpoint = Self {
            workspace_id,
            files,
            incomplete: boolean("incomplete")?,
            untracked_commands: boolean("untracked_commands")?,
        };
        if checkpoint.value() != *value {
            return Err(err("unknown checkpoint fields"));
        }
        Ok(checkpoint)
    }
    pub(crate) fn belongs_to(&self, identity: &str) -> bool {
        self.workspace_id == identity
    }
    pub fn inspect(&self, workspace: &Workspace) -> io::Result<String> {
        if workspace_id(workspace)? != self.workspace_id {
            return Err(err(
                "checkpoint belongs to a different workspace directory; start a new session",
            ));
        }
        let mut changed = Vec::new();
        for (path, revision) in &self.files {
            match workspace.stamp(path) {
                Ok(current)
                    if current.as_ref().map(|s| s.identity.as_str()) == revision.as_deref() =>
                {
                    ()
                }
                _ => changed.push(path.as_str()),
            }
        }
        Ok(format!(
            "{} checkpoint file references checked; {} changed, missing, or unavailable{}. {}{}Metadata matches are not content verification. Re-read relevant files before acting on old facts. Branches share the current workspace; switching never undoes edits. KV hints are optional; KV reuse unknown until a subsequent Engine response reports it.",
            self.files.len(),
            changed.len(),
            if changed.is_empty() {
                String::new()
            } else {
                let shown: Vec<_> = changed
                    .iter()
                    .take(16)
                    .map(|path| {
                        let mut end = path.len().min(240);
                        while !path.is_char_boundary(end) {
                            end -= 1;
                        }
                        format!(
                            "{}{}",
                            &path[..end],
                            if end < path.len() { "…" } else { "" }
                        )
                    })
                    .collect();
                format!(
                    ": {}{}",
                    shown.join(", "),
                    if changed.len() > shown.len() {
                        " [more changed paths omitted]"
                    } else {
                        ""
                    }
                )
            },
            if self.incomplete {
                "Reference set is incomplete. "
            } else {
                "Only explicitly referenced paths were sampled. "
            },
            if self.untracked_commands {
                "Commands may have changed untracked files. "
            } else {
                ""
            }
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Limits;
    use crate::test_workspace::Fixture;
    fn conversation() -> Conversation {
        Conversation::from_request(json::parse(r#"{"messages":[{"role":"user","content":"inspect"},{"role":"assistant","tool_calls":[{"id":"r","type":"function","function":{"name":"workspace_read","arguments":"{\"path\":\"file\"}"}}]},{"role":"tool","tool_call_id":"r","content":"old file contents"}]}"#).unwrap(), Limits::default()).unwrap()
    }
    #[test]
    fn revision_checks_flag_changes_without_rewriting_files_or_claiming_kv() {
        let f = Fixture::new();
        f.write("file", "old");
        let w = f.workspace();
        let checkpoint = Checkpoint::capture(&w, &conversation()).unwrap();
        let restored = Checkpoint::from_value(&checkpoint.value()).unwrap();
        assert!(restored.inspect(&w).unwrap().contains("0 changed"));
        f.write("file", "new bytes");
        let report = restored.inspect(&w).unwrap();
        assert!(report.contains("1 changed"));
        assert!(report.contains("KV reuse unknown"));
        assert_eq!(f.read("file"), "new bytes");
        assert!(restored.inspect(&Fixture::new().workspace()).is_err());
    }
}
