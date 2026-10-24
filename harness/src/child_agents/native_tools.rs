//! The real coding executor, with host-owned approvals and one private journal
//! and build session per child. Scheduler scope checks always wrap this adapter.
use super::{ChildWorker, Factory, HttpChildModel, TaskId};
use crate::{
    agent, agent_wire::Message, artifacts, coding_tools, journal, project, shell, tools::Workspace,
};
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, atomic::AtomicBool};

/// The callback belongs to the parent host, never to model/project/skill input.
/// It must keep concurrent approval requests distinct and observe cancellation.
pub type ChildPreview =
    dyn Fn(TaskId, coding_tools::Preview<'_>, &AtomicBool) -> bool + Send + Sync;

#[derive(Clone)]
pub struct NativeToolHost {
    pub workspace: Arc<Workspace>,
    pub approval: shell::ApprovalPolicy,
    pub project: project::Snapshot,
    pub shell_exclusions: shell::Exclusions,
    pub shell_toolchain: Option<PathBuf>,
    pub preview: Arc<ChildPreview>,
}

/// Pair with `ToolPolicy::native()`. Unnamed tasks are read-only even though the
/// executor has the parent's native capabilities. Named scopes only narrow that
/// executor: the parent's approval policy and callbacks still gate each action.
pub fn native_http_factory(endpoint: &str, host: NativeToolHost) -> io::Result<Arc<Factory>> {
    native_http_factory_with_generation(
        endpoint,
        host,
        crate::contract::GenerationOptions::default(),
    )
}

/// CLI sampling belongs to the conversation, including its delegated requests.
/// Child-owned output quotas and explicit request options remain authoritative.
pub fn native_http_factory_with_generation(
    endpoint: &str,
    host: NativeToolHost,
    generation: crate::contract::GenerationOptions,
) -> io::Result<Arc<Factory>> {
    crate::transport::Endpoint::parse(endpoint)?;
    let endpoint = endpoint.to_owned();
    Ok(Arc::new(move |context, cancel| {
        if cancel.load(std::sync::atomic::Ordering::Acquire) {
            return Err(io::ErrorKind::Interrupted.into());
        }
        let name = journal::fresh_name().replacen("session-", "child-", 1);
        let journal = ChildJournal(Arc::new(Mutex::new(JournalState {
            workspace: Arc::clone(&host.workspace),
            name: name.clone(),
            store: None,
        })));
        Ok(ChildWorker {
            model: Box::new(HttpChildModel::new(&endpoint)?.with_generation_options(generation)),
            tools: Box::new(NativeTools {
                id: context.id,
                uncertain: false,
                line_read: None,
                host: host.clone(),
                journal: journal.clone(),
                build: shell::BuildSession::new(host.shell_toolchain.as_deref())?,
            }),
            observer: Some(Box::new(journal)),
            journal_name: Some(name),
        })
    }))
}

struct NativeTools {
    id: TaskId,
    uncertain: bool,
    line_read: Option<crate::line_edit::Read>,
    host: NativeToolHost,
    journal: ChildJournal,
    build: shell::BuildSession,
}
impl agent::ToolExecutor for NativeTools {
    fn last_side_effect_uncertain(&self) -> bool {
        self.uncertain
    }

    fn execute(
        &mut self,
        call: agent::ToolInvocation<'_>,
        cancel: &AtomicBool,
    ) -> Result<String, agent::ToolError> {
        let host = &self.host;
        let mut tools = coding_tools::CodingTools::new(
            &host.workspace,
            host.approval,
            |preview: coding_tools::Preview<'_>, cancel: &AtomicBool| {
                (host.preview)(self.id, preview, cancel)
            },
        )
        .with_project(&host.project)
        .with_shell_exclusions(host.shell_exclusions.clone())
        .with_build_session(&self.build)
        .with_artifacts(&mut self.journal);
        tools.line_read = self.line_read.take();
        let result = tools.execute(call, cancel);
        self.line_read = tools.line_read.take();
        self.uncertain = tools.last_side_effect_uncertain();
        result
    }
}

struct JournalState {
    workspace: Arc<Workspace>,
    name: String,
    store: Option<journal::Journal>,
}
#[derive(Clone)]
struct ChildJournal(Arc<Mutex<JournalState>>);
impl ChildJournal {
    fn access<T>(&self, f: impl FnOnce(&mut journal::Journal) -> io::Result<T>) -> io::Result<T> {
        let mut state = self
            .0
            .lock()
            .map_err(|_| io::Error::other("child journal lock failed"))?;
        f(state
            .store
            .as_mut()
            .ok_or_else(|| io::Error::other("child journal not initialized"))?)
    }
}
impl agent::RunObserver for ChildJournal {
    fn checkpoint(&mut self, base: &agent::Conversation, staged: &[Message]) -> io::Result<()> {
        let mut state = self
            .0
            .lock()
            .map_err(|_| io::Error::other("child journal lock failed"))?;
        if state.store.is_none() {
            state.store = Some(journal::Journal::create(
                &state.workspace,
                &state.name,
                base,
            )?);
        }
        state.store.as_mut().unwrap().checkpoint(base, staged)
    }
    fn checkpoint_reply(
        &mut self,
        base: &agent::Conversation,
        message: &Message,
        finish: &crate::agent_wire::FinishReason,
    ) -> io::Result<()> {
        self.access(|store| store.checkpoint_reply(base, message, finish))
    }
}
impl artifacts::Access for ChildJournal {
    fn check_capacity(&mut self) -> io::Result<()> {
        self.access(artifacts::Access::check_capacity)
    }
    fn retain(&mut self, text: &str) -> io::Result<String> {
        self.access(|store| store.retain(text))
    }
    fn read(&mut self, id: &str, offset: usize, length: usize, cap: usize) -> io::Result<String> {
        self.access(|store| store.read(id, offset, length, cap))
    }
}
