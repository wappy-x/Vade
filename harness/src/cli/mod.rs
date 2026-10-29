//! Interactive CLI state shared by input, worker and presentation modules.
mod steering;
pub(crate) mod admission;
pub(crate) mod answers;
pub(crate) mod approval;
pub(crate) mod args;
pub(crate) mod attachments;
pub(crate) mod commands;
pub(crate) mod delegated;
pub(crate) mod events;
pub(crate) mod help;
mod hooks;
pub(crate) mod native_setup;
pub(crate) mod ui;
pub(crate) mod worker;

use std::io;
use std::sync::{Arc, Mutex};
use vade_harness::{
    agent::{Conversation, Limits},
    journal,
    journal::Journal,
    session::Session,
    terminal::{Terminal, editor::Editor},
    tools::Workspace,
    transport,
};

use crate::{
    delegation,
    interaction::{self, State as RunState},
    native_children,
};
use args::{Config, Mode};
use events::Pending;
use native_setup::{native_conversation, prepare_resumed};

const MAX_DISPLAY_BYTES: usize = 1024 * 1024;
pub(crate) struct App {
    config: Config,
    // Admission may time out before the staged prompt can enter the journal.
    // Retain it for the user's explicit continue without repeating any tool.
    timed_out_prompt: Option<String>,
    answers: answers::Answers,
    auto_compaction: Arc<Mutex<vade_harness::context::AutoCompaction>>,
    // Prompt tokens against the window after every turn; /context shows it.
    budget: vade_harness::context::BudgetTracker,
    terminal: Terminal,
    workspace: Arc<Workspace>,
    pub(crate) session: Session,
    /// Project instruction files; read again before a turn when they change.
    pub(crate) instructions: vade_harness::instructions::Instructions,
    hooks: vade_harness::hooks::Hooks,
    // Authoritative native context never comes from Session's text-pair recall.
    // While running, the worker exclusively owns it and must return it intact.
    pub(crate) native: Option<Conversation>,
    journal: Option<Arc<Mutex<Journal>>>,
    pub(crate) approval_reply: Option<approval::ApprovalReply>,
    approval_input_closed: bool,
    // When the current approval prompt first appeared; y/n is ignored for a
    // short settle time so a key typed before reading it stays draft text.
    pub(crate) approval_shown: Option<std::time::Instant>,
    // Last journal name seen; shown while a worker holds the journal lock.
    session_name: std::cell::RefCell<String>,
    build_session: Arc<vade_harness::shell::BuildSession>,
    native_model: Option<transport::NativeModel>,
    http: Option<transport::ChatClient>,
    direct_prefix_blocked: bool,
    native_completed: u64,
    native_report: Option<String>,
    /// Latest configured check receipt, returned independently of run termination.
    check_failing: bool,
    check_not_run: bool,
    children: native_children::Children,
    context: vade_harness::contract::Context,
    pending: Option<Pending>,
    tool_started: Option<std::time::Instant>,
    // Verbose logs show the command sandbox description once, then only changes.
    sandbox_note: String,
    backend: Option<delegation::Backend>,
    delegated_id: Option<String>,
    pub(crate) input: Editor,
    attachment_completion: (String, String),
    queued: interaction::WorkQueue,
    steering_id: u64,
    steering: Option<String>,
    queue_input: bool,
    last_state: Option<RunState>,
    last_success: bool,
    queue_paused: bool,
    pub(crate) multiline: Option<String>,
    multiline_lines: usize,
    input_overflow: bool,
    pub(crate) recall: Option<usize>,
    pub(crate) draft: String,
    paste: bool,
    dirty: bool,
    // Provisional text arrived; repaint at a bounded rate rather than per delta.
    stream_dirty: bool,
    quit: bool,
    failed: bool,
    quiet: bool,
}
impl App {
    pub(crate) fn new(config: Config) -> io::Result<Self> {
        // Tests build Config directly; the binary resolves the Vade-home default.
        let state = config.state.clone().unwrap_or(vade_harness::tools::StateLocation::Project);
        let workspace = Arc::new(Workspace::open(&config.workspace, state).map_err(|error|
            startup_error("--workspace", error,
                "Choose an existing readable directory with --workspace PATH."))?);
        let build_session = Arc::new(if config.mode == Mode::Agent {
            vade_harness::shell::BuildSession::new(config.shell_toolchain.as_deref())?
        } else {
            vade_harness::shell::BuildSession::default()
        });
        let session = Session::new(
            &workspace,
            if config.mode == Mode::Agent {
                None
            } else {
                config.session.as_deref()
            },
        ).map_err(|error| startup_error("--session", error,
            "Keep the saved metadata. Choose another --session NAME or omit --session for an in-memory chat; check workspace storage permissions."))?;
        let instructions = if config.instructions && matches!(config.mode, Mode::Agent | Mode::Direct) {
            // Tests never read the developer's own user-level file.
            let home = std::env::var_os("HOME").filter(|_| !cfg!(test)).map(std::path::PathBuf::from);
            vade_harness::instructions::Instructions::load(workspace.root(), home.as_deref())
        } else {
            vade_harness::instructions::Instructions::disabled()
        };
        let (native, journal) = if config.mode == Mode::Agent {
            let (mut store, conversation) = if let Some(query) = &config.resume {
                let name = vade_harness::saved_sessions::resolve(&workspace, query, true).map_err(|error|
                    startup_error("--resume", error, "Check --workspace and the saved --resume NAME; vade sessions lists the saved IDs and names."))?;
                let (mut store, mut conversation, torn) =
                    Journal::resume(&workspace, &name, Limits::default()).map_err(|error|
                        startup_error("--resume", error,
                            "Check --workspace and the saved --resume NAME. Keep the original journal; use a new --session NAME if it is corrupt."))?;
                prepare_resumed(
                    &mut store,
                    &mut conversation,
                    &workspace,
                    config.model.as_deref().unwrap(),
                    config.children,
                    &config.child_grants,
                    Limits { max_run_time: config.max_run_time, ..Limits::default() },
                ).map_err(|error| startup_error("--resume", error,
                    "Keep the saved journal. Resume with its original model and workspace, or choose a new --session NAME."))?;
                if torn {
                    vade_harness::diagnostics::stderr_line(format_args!(
                        "vade: recovered the last complete journal record; incomplete final record discarded."
                    ));
                }
                (store, conversation)
            } else {
                let conversation = native_conversation(
                    config.model.as_deref().unwrap(),
                    &workspace,
                    &config.rules_scopes,
                    instructions.section(),
                    config.children,
                    &config.child_grants,
                    Limits { max_run_time: config.max_run_time, ..Limits::default() },
                )?;
                let name = config.session.clone().unwrap_or_else(journal::fresh_name);
                let mut store = Journal::create(&workspace, &name, &conversation).map_err(|error|
                    startup_error("--session", error,
                        "Use --resume NAME for an existing session or choose a new --session NAME. Keep existing files and check workspace storage permissions."))?;
                store.record_details(config.name.as_deref(), Some(vade_harness::saved_sessions::now_ms())).map_err(|error|
                    startup_error("--name", error, "Check workspace storage permissions."))?;
                (store, conversation)
            };
            if config.resume.is_some() && config.name.is_some() {
                store.record_details(config.name.as_deref(), None).map_err(|error|
                    startup_error("--name", error, "Keep the saved journal; check workspace storage permissions."))?;
            }
            (Some(conversation), Some(Arc::new(Mutex::new(store))))
        } else {
            (None, None)
        };
        let http = if config.mode == Mode::Direct {
            Some(
                transport::ChatClient::new(config.endpoint.as_deref().unwrap())?
                    .with_generation_options(config.generation)
                    .with_server_cancellation(),
            )
        } else {
            None
        };
        let mut terminal = Terminal::open(config.plain || config.prompt.is_some());
        if matches!(config.mode, Mode::Codex | Mode::Hermes | Mode::Agent) {
            terminal.watch_exit_signals();
        }
        let quiet = !terminal.interactive() || config.prompt.is_some();
        let approval_input_closed = config.prompt.is_some();
        let hooks = if config.mode == Mode::Agent {
            match vade_harness::hooks::Hooks::load(&workspace) {
                Ok(hooks) => hooks,
                Err(error) => {
                    terminal.note(&format!("hooks.conf ignored for this chat: {error}"))?;
                    vade_harness::hooks::Hooks::default()
                }
            }
        } else { vade_harness::hooks::Hooks::default() };
        let mut app = Self {
            answers: answers::Answers::new(config.generation),
            budget: Default::default(),
            auto_compaction: Arc::new(Mutex::new(vade_harness::context::AutoCompaction::default())),
            config,
            attachment_completion: Default::default(),
            timed_out_prompt: None,
            terminal,
            workspace,
            session,
            instructions,
            hooks,
            native,
            journal,
            approval_reply: None,
            approval_input_closed,
            approval_shown: None,
            session_name: std::cell::RefCell::new(String::new()),
            build_session,
            native_model: None,
            http,
            direct_prefix_blocked: false,
            native_completed: 0,
            native_report: None,
            check_failing: false,
            check_not_run: false,
            children: native_children::Children::default(),
            context: vade_harness::contract::Context::default(),
            pending: None,
            tool_started: None,
            sandbox_note: String::new(),
            backend: None,
            delegated_id: None,
            input: Editor::default(),
            queued: interaction::WorkQueue::default(),
            steering_id: 0,
            steering: None,
            queue_input: false,
            last_state: None,
            last_success: false,
            queue_paused: false,
            multiline: None,
            multiline_lines: 0,
            input_overflow: false,
            recall: None,
            draft: String::new(),
            paste: false,
            dirty: true,
            stream_dirty: false,
            quit: false,
            failed: false,
            quiet,
        };
        if app.config.mode == Mode::Agent && app.config.resume.is_some() {
            let conversation = app.native.take().unwrap();
            app.replace_native(conversation, vade_harness::contract::PromptBoundary::Resume)?;
        }
        app.resumed_answers();
        Ok(app)
    }
}

fn startup_error(option: &str, error: io::Error, remedy: &str) -> io::Error {
    let detail = vade_harness::terminal::escape_preview(error.to_string().trim_end_matches('.'), 4096)
        .unwrap_or_else(|_| "local initialization failed".into());
    io::Error::new(error.kind(), format!("{option}: {detail}. {remedy}"))
}

#[cfg(test)]
mod test_support;
