//! Capability masks and in-process write ownership, not an OS sandbox.
use crate::agent::{ToolError, ToolInvocation};
use crate::json::Value;
use std::collections::BTreeMap;
use std::io;

/// What a tool may do, as the child tool policy classifies it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    /// Private parent state. Children cannot receive this capability.
    ParentOnly,
    /// A boundary-enforcing, single-target tool. No shell or multi-path edits.
    Write {
        path_argument: String,
    },
    /// Native exact patch: all sources and move destinations must be leased.
    Patch,
    /// Native sandboxed command: explicitly granted, exclusive workspace write lease.
    Shell,
}

/// Host declarations must match the actual executor. Model input cannot add
/// tools or change a tool's effect classification.
#[derive(Clone, Debug)]
pub struct ToolPolicy {
    tools: BTreeMap<String, Access>,
}

impl Default for ToolPolicy {
    fn default() -> Self {
        Self {
            tools: [
                ("workspace_read".into(), Access::Read),
                ("workspace_list".into(), Access::Read),
            ]
            .into(),
        }
    }
}

impl ToolPolicy {
    /// Registration for the real native executor. Schemas alone grant nothing;
    /// unnamed children receive only the registered read capabilities.
    pub fn native() -> Self {
        Self::native_with_web(crate::web::offered())
    }

    /// Explicit launch policy, shared with the native tool schemas.
    pub fn native_with_web(web: bool) -> Self {
        let mut policy = Self::new(
            [
                ("workspace_read".into(), Access::Read),
                ("workspace_list".into(), Access::Read),
                ("tool_output_read".into(), Access::Read),
                ("skill_read".into(), Access::Read),
                ("workspace_manifest".into(), Access::Read),
                ("workspace_symbols".into(), Access::Read),
                ("workspace_search".into(), Access::Read),
                ("workspace_read_lines".into(), Access::Read),
                ("web_fetch".into(), Access::Read),
                ("web_search".into(), Access::Read),
                ("apply_patch".into(), Access::Patch),
                ("workspace_edit_lines".into(), Access::Write { path_argument: "path".into() }),
                ("workspace_shell".into(), Access::Shell),
                ("memory_save".into(), Access::ParentOnly),
                ("memory_forget".into(), Access::ParentOnly),
            ]
            .into(),
        )
        .expect("native tool policy");
        if !web {
            policy.tools.remove("web_fetch");
            policy.tools.remove("web_search");
        }
        policy
    }

    /// Only trusted host configuration may declare additional bounded tools.
    /// Executors still own symlink/hardlink checks, atomic edits and receipts.
    pub fn new(tools: BTreeMap<String, Access>) -> io::Result<Self> {
        if tools.len() > 32 || tools.iter().any(|(name, access)| {
            !identifier(name)
                || name.starts_with("child_")
                || name == "apply_patch" && *access != Access::Patch
                || name == "workspace_edit_lines" && *access != (Access::Write { path_argument: "path".into() })
                || name == "workspace_shell" && *access != Access::Shell
                || matches!(name.as_str(), "memory_save" | "memory_forget") && *access != Access::ParentOnly
                || matches!(access, Access::Write { path_argument } if !identifier(path_argument))
        }) {
            return Err(invalid());
        }
        Ok(Self { tools })
    }

    /// Validate host declarations independently of advertised schemas. Scheduler
    /// construction also checks them against the actual frozen parent schema.
    pub fn validate_scope(&self, scope: &Scope) -> io::Result<()> {
        self.validate(scope, &self.tools.keys().cloned().collect::<Vec<_>>())
    }

    pub(crate) fn validate(&self, scope: &Scope, advertised: &[String]) -> io::Result<()> {
        if scope.tools.len() > 32
            || scope.write_paths.len() > 32
            || scope
                .tools
                .iter()
                .any(|name| !self.tools.contains_key(name) || self.parent_only(name) || !advertised.contains(name))
        {
            return Err(invalid());
        }
        for path in &scope.write_paths {
            write_path(path).map_err(|_| invalid())?;
        }
        for (i, path) in scope.write_paths.iter().enumerate() {
            if scope.write_paths[..i].iter().any(|p| overlaps(p, path)) {
                return Err(invalid());
            }
        }
        if scope.write_paths.is_empty()
            && scope.tools.iter().any(|name| {
                matches!(
                    self.tools.get(name),
                    Some(Access::Write { .. } | Access::Patch)
                )
            })
        {
            return Err(invalid());
        }
        Ok(())
    }

    pub(crate) fn read_scope(&self, advertised: &[String]) -> Scope {
        Scope {
            tools: self
                .tools
                .iter()
                .filter_map(|(name, access)| {
                    (matches!(access, Access::Read) && advertised.contains(name))
                        .then_some(name.clone())
                })
                .collect(),
            write_paths: Vec::new(),
        }
    }

    pub(crate) fn targets<'a>(
        &self,
        call: &'a ToolInvocation<'_>,
    ) -> Result<Targets<'a>, ToolError> {
        match self.tools.get(&call.call.name) {
            Some(Access::Read) => Ok(Targets::Read),
            Some(Access::Write { path_argument }) => {
                let path = call
                    .arguments
                    .get(path_argument)
                    .and_then(Value::as_str)
                    .ok_or(ToolError::InvalidArguments("write path must be a string"))?;
                Ok(Targets::Paths(vec![path]))
            }
            Some(Access::Patch) => {
                let patch = call
                    .arguments
                    .get("patch")
                    .and_then(Value::as_str)
                    .ok_or(ToolError::InvalidArguments("patch must be a string"))?;
                crate::edit::touched_paths(patch)
                    .map(Targets::Paths)
                    .map_err(|_| ToolError::InvalidArguments("patch must use the canonical patch format"))
            }
            Some(Access::Shell) => Ok(Targets::Workspace),
            Some(Access::ParentOnly) | None => Err(ToolError::Denied),
        }
    }

    pub(crate) fn parent_only(&self, name: &str) -> bool {
        self.tools.get(name) == Some(&Access::ParentOnly)
    }

    pub(crate) fn shell(&self, scope: &Scope) -> bool {
        scope
            .tools
            .iter()
            .any(|name| self.tools.get(name) == Some(&Access::Shell))
    }

    pub(crate) fn conflicts(&self, a: &Scope, b: &Scope) -> bool {
        self.shell(a) && (self.shell(b) || !b.write_paths.is_empty())
            || self.shell(b) && !a.write_paths.is_empty()
            || a.write_paths
                .iter()
                .any(|a| b.write_paths.iter().any(|b| overlaps(a, b)))
    }

    pub(crate) fn check(
        &self,
        scope: &Scope,
        call: &ToolInvocation<'_>,
    ) -> Result<bool, ToolError> {
        if !scope.tools.contains(&call.call.name) {
            return Err(ToolError::Denied);
        }
        match self.targets(call)? {
            Targets::Read => Ok(false),
            Targets::Workspace => Ok(true), // named scope explicitly includes Shell
            Targets::Paths(paths) => {
                for path in paths {
                    write_path(path)?;
                    // Exact spelling prevents aliases and directory grants. All
                    // paths in a batch, including BOTH move ends, must be leased.
                    if !scope.write_paths.iter().any(|p| p == path) {
                        return Err(ToolError::Denied);
                    }
                }
                Ok(true)
            }
        }
    }
}

pub(crate) enum Targets<'a> {
    Read,
    Paths(Vec<&'a str>),
    Workspace,
}

/// The tools and exact write paths a child may use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    pub tools: Vec<String>,
    /// Canonical lexical, relative, ASCII file paths, retained until collection.
    /// A Shell tool additionally leases all workspace writes exclusively; file
    /// tools still require exact paths here. This is ownership, not OS isolation.
    pub write_paths: Vec<String>,
}

impl Scope {
    pub fn native_read_only() -> Self {
        let policy = ToolPolicy::native();
        policy.read_scope(&policy.tools.keys().cloned().collect::<Vec<_>>())
    }
}

pub(crate) fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
}

pub(crate) fn write_path(path: &str) -> Result<(), ToolError> {
    if path.is_empty() || path.len() > 1024 || !path.is_ascii() {
        return Err(ToolError::Denied);
    }
    for part in path.split('/') {
        let lower = part.to_ascii_lowercase();
        if part.is_empty()
            || part.starts_with('.')
            || !part
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
            || matches!(
                lower.as_str(),
                "credentials"
                    | "credentials.json"
                    | "secrets"
                    | "secrets.json"
                    | "secrets.yaml"
                    | "secrets.yml"
                    | "id_rsa"
                    | "id_dsa"
                    | "id_ecdsa"
                    | "id_ed25519"
                    | "authorized_keys"
                    | "known_hosts"
                    | "netrc"
            )
            || [".pem", ".key", ".p12", ".pfx", ".keystore"]
                .iter()
                .any(|s| lower.ends_with(s))
        {
            return Err(ToolError::Denied);
        }
    }
    Ok(())
}

pub(crate) fn overlaps(a: &str, b: &str) -> bool {
    let a = a.to_ascii_lowercase();
    let b = b.to_ascii_lowercase();
    a == b
        || a.strip_prefix(&b).is_some_and(|tail| tail.starts_with('/'))
        || b.strip_prefix(&a).is_some_and(|tail| tail.starts_with('/'))
}

fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "invalid child tool scope")
}
