//! Native CLI ownership of the current/last activation's bounded children.
//! The UI has cancellation and metadata access; only the worker collects results.
use std::collections::BTreeMap;
use std::io;
use std::sync::{Arc, atomic::AtomicBool};
use vade_harness::{
    agent::Conversation,
    child_agents::{
        self, ChildControl, ChildView, Scheduler, StablePrefix, TaskResult, TaskSnapshot,
    },
};

#[derive(Default)]
pub struct Children {
    view: Option<ChildView>,
    control: Option<ChildControl>,
    results: Vec<TaskResult>,
    revision: u64,
}

impl Children {
    /// This validates configuration and starts a local monitor only. HTTP stays
    /// on the child worker; setup cannot block the input thread on an endpoint.
    pub fn prepare(
        &mut self,
        parent: &Conversation,
        endpoint: &str,
        host: child_agents::NativeToolHost,
        generation: vade_harness::contract::GenerationOptions,
        grants: BTreeMap<String, child_agents::Scope>,
        cancel: Arc<AtomicBool>,
    ) -> io::Result<Scheduler> {
        let scheduler = Scheduler::new(
            StablePrefix::from_parent(parent)?,
            child_agents::Config {
                policy: child_agents::ToolPolicy::native(),
                grants,
                ..child_agents::Config::default()
            },
            cancel,
            child_agents::native_http_factory_with_generation(endpoint, host, generation)?,
        )?;
        let view = scheduler.view();
        view.wake_current_thread();
        self.view = Some(view);
        self.control = Some(scheduler.control());
        self.results.clear();
        self.revision = 0;
        Ok(scheduler)
    }

    pub fn finish(&mut self, results: Vec<TaskResult>) {
        // Closed child transcripts remain owned by this parent activation and
        // are never reconstructed from displayed text or replayed as tools.
        self.results = results;
        self.control = None;
    }

    pub fn snapshots(&self) -> Vec<TaskSnapshot> {
        self.view
            .as_ref()
            .map_or_else(Vec::new, ChildView::snapshots)
    }

    pub fn changed(&mut self) -> bool {
        let revision = self.view.as_ref().map_or(0, ChildView::revision);
        let changed = revision != self.revision;
        self.revision = revision;
        changed
    }

    pub fn cancel(&self, target: &str) -> String {
        let Some(control) = &self.control else {
            return "No active child scheduler; completed child outcomes are retained.".into();
        };
        if target == "all" {
            let count = control.cancel_all();
            return format!(
                "Child cancellation requested for {count} active task(s); further spawns closed for this activation. Remote stop unconfirmed."
            );
        }
        match control.cancel(target) {
            Ok(true) => format!(
                "Child cancellation requested: {target}. Waiting for local cleanup; remote stop unconfirmed."
            ),
            Ok(false) => {
                "Child already stopping or finished; its actual outcome is retained.".into()
            }
            Err(_) => "Unknown child task ID in this activation; use /children to list IDs.".into(),
        }
    }

    pub fn retained_results(&self) -> usize {
        self.results.len()
    }
}

/// Only command-line host configuration calls this; model requests can select
/// a name but cannot create or extend a scope.
pub fn add_grant(
    grants: &mut BTreeMap<String, child_agents::Scope>,
    name: &str,
    path: Option<&str>,
) -> Result<(), &'static str> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
    {
        return Err("invalid child scope name");
    }
    if !grants.contains_key(name) && grants.len() >= 32 {
        return Err("at most 32 child scopes");
    }
    let mut scope = grants
        .get(name)
        .cloned()
        .unwrap_or_else(child_agents::Scope::native_read_only);
    let tool = if let Some(path) = path {
        scope.write_paths.push(path.into());
        "apply_patch"
    } else {
        "workspace_shell"
    };
    if !scope.tools.iter().any(|name| name == tool) {
        scope.tools.push(tool.into());
    }
    if path.is_some() && !scope.tools.iter().any(|name| name == "workspace_edit_lines") {
        scope.tools.push("workspace_edit_lines".into());
    }
    child_agents::ToolPolicy::native()
        .validate_scope(&scope)
        .map_err(|_| "invalid or overlapping child write paths (use exact relative ASCII files)")?;
    scope.write_paths.sort();
    scope.tools.sort();
    grants.insert(name.into(), scope);
    Ok(())
}

pub fn describe_grants(grants: &BTreeMap<String, child_agents::Scope>) -> String {
    if grants.is_empty() {
        return "No named child scopes. Unnamed children are read-only.".into();
    }
    let mut text = String::from(
        "Host-defined child scopes (select by child_spawn grant; approvals still apply):",
    );
    for (name, scope) in grants {
        text.push_str(&format!(
            "\n{name}: edits [{}]; shell {}",
            scope.write_paths.join(", "),
            if scope.tools.iter().any(|name| name == "workspace_shell") {
                "explicitly granted, exclusive workspace write lease"
            } else {
                "denied"
            }
        ));
    }
    text
}
