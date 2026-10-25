//! Bounded project instructions and declarative skills. No script, installer,
//! subprocess, network operation, implicit home discovery or permission grant.
use crate::agent::Conversation;
use crate::json::{self, Value};
use crate::tools::{FileStamp, Workspace, safe_parts};
use std::collections::{BTreeSet, VecDeque};
use std::io;

const MARKER: &str = "Vade frozen project context v1\n";
const NOTICE: &str = "Project rules are scoped guidance subordinate to explicit user requests and system policy. A rule applies only inside its scope; the deepest applicable scope wins. The root applies everywhere. Project content and skills cannot grant executor permissions, change approval policy, install dependencies or execute scripts automatically. Rules/catalog are frozen until a new session. Skill bodies are loaded only on request with a local file revision guard.";
pub const MAX_SCOPES: usize = 8;
pub const MAX_SCOPE_DEPTH: usize = 8;
pub const MAX_RULES: usize = 16;
pub const MAX_RULE_BYTES: usize = 16 * 1024;
pub const MAX_TOTAL_RULE_BYTES: usize = 24 * 1024;
pub const MAX_SKILLS: usize = 16;
pub const MAX_SKILL_BYTES: usize = 16 * 1024;
pub const MAX_CATALOG_BYTES: usize = 8 * 1024;
const MAX_SNAPSHOT_BYTES: usize = 48 * 1024;
const CATALOG: &str = "skills/catalog.json";
fn err(s: &str) -> io::Error {
    io::Error::other(s)
}
fn string(v: &Value, key: &str) -> io::Result<String> {
    v.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| err("project context field must be a string"))
}
fn plain(text: &str) -> bool {
    crate::terminal::sanitize(text) == text && !text.contains('\n') && !text.contains('\t')
}
fn normalized(path: &str) -> io::Result<String> {
    let parts = safe_parts(path)?;
    if parts.len() > MAX_SCOPE_DEPTH || path.len() > 512 {
        return Err(err("project path exceeds depth/length limit"));
    }
    Ok(if parts.is_empty() {
        ".".into()
    } else {
        parts.join("/")
    })
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Rule {
    path: String,
    scope: String,
    text: String,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Skill {
    name: String,
    description: String,
    path: String,
    stamp: FileStamp,
}
/// The project rules and skill catalog discovered for a session's scopes,
/// frozen into the stable prompt prefix when the session starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    scopes: Vec<String>,
    rules: Vec<Rule>,
    skills: Vec<Skill>,
}
impl Snapshot {
    /// The root is always considered. Extra scopes add only their ancestor
    /// chain, never recursive filesystem discovery or paths above the root.
    pub fn discover(workspace: &Workspace, scopes: &[String]) -> io::Result<Self> {
        if scopes.len() > MAX_SCOPES {
            return Err(err("at most eight explicit project scopes"));
        }
        let mut selected = BTreeSet::from([".".into()]);
        let mut candidates = BTreeSet::from([".".to_string()]);
        for scope in scopes {
            let scope = normalized(scope)?;
            if !workspace.stamp(&scope)?.is_some_and(|s| s.directory) {
                return Err(err(
                    "instruction scope must be an existing workspace directory",
                ));
            }
            selected.insert(scope.clone());
            if scope != "." {
                let mut parents = String::new();
                for part in scope.split('/') {
                    if !parents.is_empty() {
                        parents.push('/');
                    }
                    parents.push_str(part);
                    candidates.insert(parents.clone());
                }
            }
        }
        let mut candidates: Vec<_> = candidates.into_iter().collect();
        candidates.sort_by_key(|p| (if p == "." { 0 } else { p.split('/').count() }, p.clone()));
        let mut rules = Vec::new();
        let mut total = 0;
        for scope in candidates {
            let path = if scope == "." {
                "AGENTS.md".into()
            } else {
                format!("{scope}/AGENTS.md")
            };
            let text = workspace.read_optional_bounded(&path, MAX_RULE_BYTES).map_err(|e| {
                err(&format!("project rules {path}: {e}; each rules file may hold up to 16 KiB"))
            })?;
            if let Some(text) = text {
                total += text.len();
                if rules.len() >= MAX_RULES || total > MAX_TOTAL_RULE_BYTES {
                    return Err(err(
                        "project rules exceed 16 files/24 KiB; choose fewer scopes",
                    ));
                }
                rules.push(Rule { path, scope, text });
            }
        }
        let mut skills = Vec::new();
        if let Some(stamp) = workspace.stamp("skills")? {
            if !stamp.directory {
                return Err(err("skills must be a directory"));
            }
            if let Some(text) = workspace.read_optional_bounded(CATALOG, MAX_CATALOG_BYTES)? {
                let catalog = json::parse(&text).map_err(|_| err("invalid skills/catalog.json"))?;
                let Value::Object(fields) = &catalog else {
                    return Err(err("skill catalog must be an object"));
                };
                if fields.len() != 1 {
                    return Err(err("skill catalog accepts only a skills array"));
                }
                let entries = catalog
                    .get("skills")
                    .and_then(Value::as_array)
                    .ok_or_else(|| err("skill catalog needs skills array"))?;
                if entries.len() > MAX_SKILLS {
                    return Err(err("skill catalog exceeds 16 entries"));
                }
                for entry in entries {
                    let Value::Object(fields) = entry else {
                        return Err(err("skill entry must be an object"));
                    };
                    if fields.len() != 3 {
                        return Err(err(
                            "skill entry needs only name, description, path; scripts are unsupported",
                        ));
                    }
                    let name = string(entry, "name")?;
                    let description = string(entry, "description")?;
                    let path = normalized(&string(entry, "path")?)?;
                    validate_skill_descriptor(&name, &description, &path)?;
                    let stamp = workspace
                        .stamp(&path)?
                        .ok_or_else(|| err("skill body missing"))?;
                    if stamp.directory || stamp.bytes > MAX_SKILL_BYTES as u64 {
                        return Err(err("skill body exceeds 16 KiB or is not a regular file"));
                    }
                    skills.push(Skill {
                        name,
                        description,
                        path,
                        stamp,
                    });
                }
            }
        }
        skills.sort_by(|a, b| a.name.cmp(&b.name));
        if skills.windows(2).any(|s| s[0].name == s[1].name) {
            return Err(err("duplicate skill name"));
        }
        let mut scopes: Vec<_> = selected.into_iter().collect();
        scopes.sort_by_key(|p| (p != ".", p.clone()));
        let snapshot = Self {
            scopes,
            rules,
            skills,
        };
        snapshot.message()?; // enforce the frozen serialized prefix bound now
        Ok(snapshot)
    }
    fn value(&self) -> Value {
        Value::Object(vec![
            ("notice".into(), Value::String(NOTICE.into())),
            (
                "selected_scopes".into(),
                Value::Array(self.scopes.iter().cloned().map(Value::String).collect()),
            ),
            (
                "rules".into(),
                Value::Array(
                    self.rules
                        .iter()
                        .map(|r| {
                            Value::Object(vec![
                                ("path".into(), Value::String(r.path.clone())),
                                ("scope".into(), Value::String(r.scope.clone())),
                                ("text".into(), Value::String(r.text.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("catalog_source".into(), Value::String(CATALOG.into())),
            (
                "skills".into(),
                Value::Array(
                    self.skills
                        .iter()
                        .map(|s| {
                            Value::Object(vec![
                                ("name".into(), Value::String(s.name.clone())),
                                ("description".into(), Value::String(s.description.clone())),
                                ("path".into(), Value::String(s.path.clone())),
                                ("bytes".into(), Value::Number(s.stamp.bytes.to_string())),
                                (
                                    "local_revision".into(),
                                    Value::String(s.stamp.identity.clone()),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
    pub fn message(&self) -> io::Result<Value> {
        let text = format!("{MARKER}{}", json::stringify(&self.value())?);
        if text.len() > MAX_SNAPSHOT_BYTES {
            return Err(err(
                "project context exceeds 48 KiB serialized snapshot limit",
            ));
        }
        Ok(Value::Object(vec![
            ("role".into(), Value::String("developer".into())),
            ("content".into(), Value::String(text)),
        ]))
    }
    pub fn from_conversation(c: &Conversation) -> io::Result<Self> {
        let mut found = None;
        for m in &c.messages()[..c.prefix_len()] {
            if let Some(text) = m
                .value()
                .get("content")
                .and_then(Value::as_str)
                .and_then(|s| s.strip_prefix(MARKER))
            {
                if found.is_some() || text.len() > MAX_SNAPSHOT_BYTES {
                    return Err(err("duplicate or oversized frozen project context"));
                }
                found = Some(Self::from_value(json::parse(text)?)?);
            }
        }
        found.ok_or_else(|| err("session has no frozen project context; start a new session"))
    }
    fn from_value(v: Value) -> io::Result<Self> {
        if string(&v, "notice")? != NOTICE || string(&v, "catalog_source")? != CATALOG {
            return Err(err("unsupported frozen project context"));
        }
        let array = |key| {
            v.get(key)
                .and_then(Value::as_array)
                .ok_or_else(|| err("invalid project array"))
        };
        let mut scopes = Vec::new();
        for scope in array("selected_scopes")? {
            let path = scope.as_str().ok_or_else(|| err("invalid scope"))?;
            if scopes.len() > MAX_SCOPES || normalized(path)? != path {
                return Err(err("invalid or excessive scopes"));
            }
            scopes.push(path.into());
        }
        if scopes.first().map(String::as_str) != Some(".")
            || scopes[1..].windows(2).any(|s| s[0] >= s[1])
        {
            return Err(err("noncanonical scopes"));
        }
        let mut rules = Vec::new();
        let mut total = 0;
        for r in array("rules")? {
            let path = string(r, "path")?;
            let scope = string(r, "scope")?;
            let text = string(r, "text")?;
            let expected = if scope == "." {
                "AGENTS.md".into()
            } else {
                format!("{scope}/AGENTS.md")
            };
            total += text.len();
            if normalized(&scope)? != scope
                || path != expected
                || text.len() > MAX_RULE_BYTES
                || total > MAX_TOTAL_RULE_BYTES
                || rules.len() >= MAX_RULES
            {
                return Err(err("invalid or excessive frozen rules"));
            }
            rules.push(Rule { path, scope, text });
        }
        rules.sort_by_key(|r| {
            (
                if r.scope == "." {
                    0
                } else {
                    r.scope.split('/').count()
                },
                r.scope.clone(),
            )
        });
        if rules.windows(2).any(|r| r[0].scope == r[1].scope)
            || rules.iter().any(|r| {
                r.scope != "."
                    && !scopes
                        .iter()
                        .any(|s| *s == r.scope || s.starts_with(&format!("{}/", r.scope)))
            })
        {
            return Err(err("duplicate or unselected frozen rule scope"));
        }
        let mut skills = Vec::new();
        for s in array("skills")? {
            let name = string(s, "name")?;
            let description = string(s, "description")?;
            let path = string(s, "path")?;
            validate_skill_descriptor(&name, &description, &path)?;
            let bytes = match s.get("bytes") {
                Some(Value::Number(n)) => {
                    n.parse::<u64>().map_err(|_| err("invalid skill bytes"))?
                }
                _ => return Err(err("missing skill bytes")),
            };
            let identity = string(s, "local_revision")?;
            if bytes > MAX_SKILL_BYTES as u64
                || identity.len() > 200
                || !identity
                    .bytes()
                    .all(|b| b.is_ascii_digit() || b == b':' || b == b'-')
                || skills.len() >= MAX_SKILLS
            {
                return Err(err("invalid or excessive skill revision"));
            }
            skills.push(Skill {
                name,
                description,
                path,
                stamp: FileStamp {
                    directory: false,
                    bytes,
                    identity,
                },
            });
        }
        if skills.windows(2).any(|s| s[0].name >= s[1].name) {
            return Err(err("noncanonical skill names"));
        }
        let result = Self {
            scopes,
            rules,
            skills,
        };
        if result.value() != v {
            return Err(err("noncanonical frozen project context"));
        }
        Ok(result)
    }
    pub fn load_skill(&self, workspace: &Workspace, name: &str) -> io::Result<String> {
        let skill = self
            .skills
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| err("skill is not in the frozen catalog"))?;
        let (text, stamp) = workspace
            .read_snapshot_bounded(&skill.path, MAX_SKILL_BYTES)?
            .ok_or_else(|| err("skill body missing; use /new to refresh"))?;
        if stamp != skill.stamp {
            return Err(err(
                "skill revision changed; use /new to refresh rules/catalog, then explicitly load it again",
            ));
        }
        Ok(format!(
            "Skill data loaded on request: {} · source {} · local metadata revision {}. Exact loaded bytes follow; no scripts were executed and tool permissions are unchanged.\n{}",
            skill.name, skill.path, skill.stamp.identity, text
        ))
    }
    pub fn inspect(&self) -> String {
        let mut text = format!(
            "Frozen project context · selected scopes: {}. /new reloads; resume preserves this snapshot.\n",
            self.scopes.join(", ")
        );
        for r in &self.rules {
            text.push_str(&format!(
                "rule {} · applies under {} · {} bytes · included by selected ancestor chain\n",
                r.path,
                r.scope,
                r.text.len()
            ));
        }
        if self.rules.is_empty() {
            text.push_str("No AGENTS.md files in the selected ancestor chains.\n");
        }
        for s in &self.skills {
            text.push_str(&format!(
                "skill {} · {} · {} bytes · body lazy · revision {}\n",
                s.name, s.description, s.stamp.bytes, s.stamp.identity
            ));
        }
        text.push_str("/project-rules displays exact frozen rule text locally. /skill NAME loads a body locally; the model uses skill_read. Repository views are on-demand, not inserted into the frozen prefix.");
        text
    }
    pub fn inspect_rules(&self) -> String {
        let mut text = String::new();
        for r in &self.rules {
            text.push_str(&format!(
                "--- {} (scope {}) ---\n{}\n",
                r.path, r.scope, r.text
            ));
        }
        if text.is_empty() {
            text.push_str("No frozen project rules.");
        }
        text
    }
}
fn validate_skill_descriptor(name: &str, description: &str, path: &str) -> io::Result<()> {
    if !crate::journal::valid_name(name)
        || description.is_empty()
        || description.len() > 240
        || !plain(description)
        || !path.starts_with("skills/")
        || !path.ends_with("/SKILL.md")
        || normalized(path)? != path
    {
        return Err(err(
            "skill needs a unique 1–48 ASCII name, plain description ≤240 bytes, and a visible skills/.../SKILL.md path",
        ));
    }
    Ok(())
}

/// A lexical repository view: names only, explicit incomplete markers, no AST
/// claims, hidden traversal, dependency scan, or automatic prompt insertion.
pub fn manifest(workspace: &Workspace, path: &str, depth: usize) -> io::Result<String> {
    if depth > 3 {
        return Err(err("manifest depth must be 0–3"));
    }
    let root = normalized(path)?;
    if root
        .split('/')
        .any(|p| ["target", "node_modules", "vendor"].contains(&p))
    {
        return Err(err("build/dependency trees are excluded from the manifest"));
    }
    let mut queue = VecDeque::from([(root, 0)]);
    let mut out = String::from(
        "Bounded repository names (no semantic index; target/node_modules/vendor and hidden paths excluded):\n",
    );
    let mut entries = 0;
    let mut dirs = 0;
    while let Some((path, level)) = queue.pop_front() {
        dirs += 1;
        if dirs > 32 {
            out.push_str("[directory limit reached]\n");
            break;
        }
        let names = workspace.list(&path)?;
        for name in names {
            if name == "[listing limit reached]" {
                out.push_str("[listing incomplete]\n");
                continue;
            }
            entries += 1;
            if entries > 256 || out.len() > 48 * 1024 {
                out.push_str("[manifest limit reached]\n");
                return Ok(out);
            }
            if ["target", "node_modules", "vendor"].contains(&name.as_str()) {
                continue;
            }
            let child = if path == "." {
                name
            } else {
                format!("{path}/{name}")
            };
            match workspace.stamp(&child) {
                Ok(Some(stamp)) => {
                    out.push_str(&child);
                    if stamp.directory {
                        out.push('/');
                        if level < depth {
                            queue.push_back((child, level + 1));
                        }
                    }
                    out.push('\n');
                }
                _ => out.push_str(&format!("{child} [unavailable; not followed]\n")),
            }
        }
    }
    Ok(out)
}
/// Up to 64 lexical declaration hints from one workspace file, with line
/// numbers; a string scan, not a parser.
pub fn symbols(workspace: &Workspace, path: &str) -> io::Result<String> {
    let text = workspace.read(path)?;
    let mut out = String::from(
        "Lexical line hints only; comments, strings and language syntax can cause false positives/misses:\n",
    );
    let mut matches = 0;
    for (index, line) in text.lines().enumerate() {
        let mut candidate = line.trim_start();
        for prefix in ["pub ", "export ", "async "] {
            candidate = candidate.strip_prefix(prefix).unwrap_or(candidate);
        }
        if [
            "fn ",
            "struct ",
            "enum ",
            "trait ",
            "impl ",
            "class ",
            "def ",
            "function ",
            "interface ",
            "type ",
        ]
        .iter()
        .any(|p| candidate.starts_with(p))
        {
            if matches >= 64 {
                out.push_str("[symbol hint limit reached]\n");
                break;
            }
            let mut end = line.len().min(240);
            while !line.is_char_boundary(end) {
                end -= 1;
            }
            out.push_str(&format!(
                "{}: {}{}\n",
                index + 1,
                &line[..end],
                if end < line.len() {
                    " [line excerpt]"
                } else {
                    ""
                }
            ));
            matches += 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Limits;
    use crate::test_workspace::Fixture;
    fn prepare() -> Fixture {
        let f = Fixture::new();
        for path in ["src/api", "-scope", "skills/a", "skills/z"] {
            std::fs::create_dir_all(f.0.join(path)).unwrap();
        }
        f.write("AGENTS.md", "root rules");
        f.write("src/AGENTS.md", "src rules apply only here");
        f.write("src/api/AGENTS.md", "api overrides src here");
        f.write(
            "skills/a/SKILL.md",
            "# A\nDo not run this illustrative script: touch never-created\n",
        );
        f.write("skills/z/SKILL.md", "# Z\nAnother skill body\n");
        f.write(CATALOG, r#"{"skills":[{"name":"z","description":"z description","path":"skills/z/SKILL.md"},{"name":"a","description":"a description","path":"skills/a/SKILL.md"}]}"#);
        f
    }
    fn c(snapshot: &Snapshot) -> Conversation {
        Conversation::from_request(
            Value::Object(vec![
                ("model".into(), Value::String("mock".into())),
                (
                    "messages".into(),
                    Value::Array(vec![snapshot.message().unwrap()]),
                ),
            ]),
            Limits::default(),
        )
        .unwrap()
    }
    #[test]
    fn scope_and_catalog_order_are_deterministic_with_roundtrip_and_no_autorun() {
        let f = prepare();
        let w = f.workspace();
        let a = Snapshot::discover(&w, &["src/api".into(), "-scope".into()]).unwrap();
        assert_eq!(
            a.rules.iter().map(|r| r.scope.as_str()).collect::<Vec<_>>(),
            [".", "src", "src/api"]
        );
        let b = Snapshot::discover(&w, &["-scope".into(), "src/api".into(), "src".into()]).unwrap();
        // Explicit selected scopes differ, but effective precedence/order does not.
        assert_eq!(a.rules, b.rules);
        let Value::Object(mut catalog) = json::parse(&f.read(CATALOG)).unwrap() else {
            panic!()
        };
        let Value::Array(entries) = &mut catalog[0].1 else {
            panic!()
        };
        entries.reverse();
        f.write(CATALOG, &json::stringify(&Value::Object(catalog)).unwrap());
        let reordered = Snapshot::discover(&w, &["-scope".into(), "src/api".into()]).unwrap();
        assert_eq!(a.message().unwrap(), reordered.message().unwrap());
        assert_eq!(a, Snapshot::from_conversation(&c(&a)).unwrap());
        assert!(
            !json::stringify(&a.message().unwrap())
                .unwrap()
                .contains("touch never-created")
        );
        assert!(
            a.load_skill(&w, "a")
                .unwrap()
                .contains("touch never-created")
        );
        assert!(!f.0.join("never-created").exists());
        assert_eq!(Snapshot::discover(&w, &[]).unwrap().rules.len(), 1);
    }
    #[test]
    fn frozen_rules_survive_changes_and_lazy_skill_revision_changes_fail_closed() {
        let f = prepare();
        let w = f.workspace();
        let snapshot = Snapshot::discover(&w, &["src/api".into()]).unwrap();
        let stored = c(&snapshot);
        f.write("AGENTS.md", "changed root");
        f.write(CATALOG, r#"{"skills":[]}"#);
        f.write(
            "skills/a/SKILL.md",
            "new body must not silently replace frozen revision",
        );
        let restored = Snapshot::from_conversation(&stored).unwrap();
        assert_eq!(restored, snapshot);
        assert!(restored.inspect_rules().contains("root rules"));
        assert!(!restored.inspect_rules().contains("changed root"));
        assert!(
            restored
                .load_skill(&w, "a")
                .unwrap_err()
                .to_string()
                .contains("revision changed")
        );
        assert!(restored.load_skill(&w, "unlisted").is_err());
        let fresh = Snapshot::discover(&w, &[]).unwrap();
        assert!(fresh.inspect_rules().contains("changed root"));
        assert!(fresh.skills.is_empty());
    }
    #[test]
    fn skill_body_is_not_read_during_discovery() {
        let f = prepare();
        std::fs::write(f.0.join("skills/a/SKILL.md"), [255u8, 254]).unwrap();
        let w = f.workspace();
        let snapshot = Snapshot::discover(&w, &[]).unwrap(); // metadata only
        assert!(snapshot.load_skill(&w, "a").is_err()); // UTF-8 checked lazily
    }
    #[test]
    fn oversized_duplicate_and_executable_catalog_entries_are_rejected() {
        let f = prepare();
        let w = f.workspace();
        f.write("AGENTS.md", &"x".repeat(MAX_RULE_BYTES + 1));
        assert!(Snapshot::discover(&w, &[]).is_err());
        f.write("AGENTS.md", "safe");
        f.write("skills/a/SKILL.md", &"x".repeat(MAX_SKILL_BYTES + 1));
        assert!(Snapshot::discover(&w, &[]).is_err());
        f.write("skills/a/SKILL.md", "safe");
        for catalog in [
            r#"{"skills":[{"name":"a","description":"a","path":"skills/a/SKILL.md"},{"name":"a","description":"duplicate","path":"skills/z/SKILL.md"}]}"#,
            r#"{"skills":[{"name":"a","description":"a","path":"skills/a/SKILL.md","script":"touch forbidden"}]}"#,
            r#"{"skills":[{"name":"a","description":"a","path":"../SKILL.md"}]}"#,
            r#"{"skills":[{"name":"a","description":"a","path":"skills/.private/SKILL.md"}]}"#,
        ] {
            f.write(CATALOG, catalog);
            assert!(Snapshot::discover(&w, &[]).is_err());
        }
        f.write(CATALOG, &" ".repeat(MAX_CATALOG_BYTES + 1));
        assert!(Snapshot::discover(&w, &[]).is_err());
        assert!(Snapshot::discover(&w, &vec!["src".into(); MAX_SCOPES + 1]).is_err());
        assert!(Snapshot::discover(&w, &["../outside".into()]).is_err());
        assert!(!f.0.join("forbidden").exists());
    }
    #[test]
    fn project_boundaries_reject_rule_skill_and_scope_redirects() {
        use std::os::unix::fs::symlink;
        let f = prepare();
        let outside = prepare();
        let w = f.workspace();
        std::fs::remove_file(f.0.join("AGENTS.md")).unwrap();
        symlink(outside.0.join("AGENTS.md"), f.0.join("AGENTS.md")).unwrap();
        assert!(Snapshot::discover(&w, &[]).is_err());
        std::fs::remove_file(f.0.join("AGENTS.md")).unwrap();
        f.write("AGENTS.md", "safe");
        std::fs::remove_file(f.0.join("skills/a/SKILL.md")).unwrap();
        std::fs::hard_link(
            outside.0.join("skills/a/SKILL.md"),
            f.0.join("skills/a/SKILL.md"),
        )
        .unwrap();
        assert!(Snapshot::discover(&w, &[]).is_err());
        symlink(outside.0.join("src"), f.0.join("escape")).unwrap();
        assert!(Snapshot::discover(&w, &["escape/api".into()]).is_err());
    }
    #[test]
    fn repository_views_are_bounded_and_label_lexical_limits() {
        let f = prepare();
        std::fs::create_dir(f.0.join("target")).unwrap();
        f.write("target/never-list", "artifact");
        f.write(".private", "private");
        f.write(
            "src/code.rs",
            &("pub fn example() {}\n// fn comment() {}\n".repeat(100)),
        );
        let w = f.workspace();
        let manifest = manifest(&w, ".", 3).unwrap();
        assert!(manifest.contains("src/code.rs"));
        assert!(!manifest.contains("never-list"));
        assert!(!manifest.contains(".private"));
        assert!(super::manifest(&w, ".", 4).is_err());
        let symbols = symbols(&w, "src/code.rs").unwrap();
        assert!(symbols.starts_with("Lexical line hints only"));
        assert_eq!(symbols.matches("pub fn example").count(), 64);
        assert!(symbols.contains("limit reached"));
        for i in 0..300 {
            f.write(&format!("entry-{i}"), "");
        }
        let large = super::manifest(&w, ".", 3).unwrap();
        assert!(large.len() < 65536);
        assert!(large.contains("incomplete") || large.contains("limit reached"));
    }
}
