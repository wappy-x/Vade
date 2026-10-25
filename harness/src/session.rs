//! In-memory conversation and optional content-free, private session metadata.
use crate::protocol::Message;
use crate::tools::Workspace;
use std::io;

pub const SYSTEM_PROMPT: &str = "You are Vade, a clear and careful coding assistant. Answer the user's request directly. Be truthful about what you have and have not done. You cannot execute tools; workspace commands are explicitly controlled by the user. Do not claim file changes, tests, or network actions you have not performed.";
pub const MAX_PROMPT_BYTES: usize = 64 * 1024;
pub const MAX_HISTORY_BYTES: usize = 256 * 1024;
pub const MAX_HISTORY_TURNS: usize = 32;
pub const MAX_INPUT_HISTORY_BYTES: usize = 512 * 1024;
pub const MAX_INPUT_HISTORY_ENTRIES: usize = 500;

/// The only thing a direct-chat session saves: its completed-turn count, never
/// the conversation.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct Metadata {
    pub completed_turns: u64,
}

/// A direct-chat conversation kept in memory within byte and turn limits,
/// with optional private metadata on disk.
pub struct Session {
    history: Vec<Message>,
    bytes: usize,
    pub metadata: Metadata,
    pub pruned_turns: usize,
    store: Option<Store>,
}

impl Session {
    pub fn new(workspace: &Workspace, name: Option<&str>) -> io::Result<Self> {
        let store = name.map(|name| Store::open(workspace, name)).transpose()?;
        let metadata = store
            .as_ref()
            .map(Store::load)
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            history: Vec::new(),
            bytes: 0,
            metadata,
            pruned_turns: 0,
            store,
        })
    }
    pub fn history(&self) -> &[Message] {
        &self.history
    }
    pub fn history_bytes(&self) -> usize {
        self.bytes
    }
    pub fn turns(&self) -> usize {
        self.history.len() / 2
    }
    pub fn persistent(&self) -> bool {
        self.store.is_some()
    }

    /// Retain explicitly interrupted text without counting a completed turn.
    pub fn interrupted(&mut self, user: String, assistant: String) -> io::Result<bool> {
        self.record(user, assistant)
    }
    /// Store only complete exchanges. Oversized exchanges never become a partial
    /// prompt; the caller can report that they were excluded from future context.
    pub fn complete(&mut self, user: String, assistant: String) -> io::Result<bool> {
        self.metadata.completed_turns = self.metadata.completed_turns.saturating_add(1);
        self.record(user, assistant)
    }
    fn record(&mut self, user: String, assistant: String) -> io::Result<bool> {
        let bytes = user.len().saturating_add(assistant.len());
        let retained = bytes <= MAX_HISTORY_BYTES;
        if retained {
            while self.turns() >= MAX_HISTORY_TURNS || self.bytes + bytes > MAX_HISTORY_BYTES {
                self.bytes -= self.history[0].content.len() + self.history[1].content.len();
                self.history.drain(..2);
                self.pruned_turns += 1;
            }
            self.bytes += bytes;
            self.history.push(Message {
                role: "user".into(),
                content: user,
            });
            self.history.push(Message {
                role: "assistant".into(),
                content: assistant,
            });
        }
        if let Some(store) = &self.store {
            store.save(&self.metadata)?;
        }
        Ok(retained)
    }
    pub fn clear(&mut self) {
        self.history.clear();
        self.bytes = 0;
        self.pruned_turns = 0;
    }
    pub fn undo(&mut self) -> bool {
        if self.history.is_empty() {
            return false;
        }
        for _ in 0..2 {
            self.bytes -= self.history.pop().unwrap().content.len();
        }
        true
    }
}

struct Store {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    dir: crate::tools::anchored::Dir,
    filename: String,
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    _lock: crate::tools::anchored::Lock,
}
impl Store {
    fn open(workspace: &Workspace, name: &str) -> io::Result<Self> {
        if name.is_empty()
            || name.len() > 48
            || !name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        {
            return Err(io::Error::other(
                "session name must be 1–48 ASCII letters, digits, '-' or '_'",
            ));
        }
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            let dir = workspace.session_dir()?;
            let lock = dir.lock(&format!("{name}.lock"))?;
            Ok(Self {
                dir,
                filename: format!("{name}.state"),
                _lock: lock,
            })
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = workspace;
            Err(io::Error::other(
                "private session metadata requires macOS or Linux",
            ))
        }
    }
    fn load(&self) -> io::Result<Metadata> {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            use std::io::Read;
            let Some(file) = self.dir.optional_file(&self.filename)? else {
                return Ok(Metadata::default());
            };
            let mut bytes = Vec::new();
            file.take(257).read_to_end(&mut bytes)?;
            let text = std::str::from_utf8(&bytes).map_err(|_| invalid_metadata())?;
            parse_metadata(text)
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Err(io::Error::other(
                "private session metadata requires macOS or Linux",
            ))
        }
    }
    fn save(&self, metadata: &Metadata) -> io::Result<()> {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            self.dir.write_atomic(
                &self.filename,
                format!(
                    "vade-session-v1\ncompleted_turns={}\n",
                    metadata.completed_turns
                )
                .as_bytes(),
            )
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = metadata;
            Err(io::Error::other(
                "private session metadata requires macOS or Linux",
            ))
        }
    }
}
fn invalid_metadata() -> io::Error {
    io::Error::other("invalid session metadata; choose a new session name")
}
fn parse_metadata(text: &str) -> io::Result<Metadata> {
    let value = text
        .strip_prefix("vade-session-v1\ncompleted_turns=")
        .and_then(|s| s.strip_suffix('\n'))
        .ok_or_else(invalid_metadata)?;
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid_metadata());
    }
    Ok(Metadata {
        completed_turns: value.parse().map_err(|_| invalid_metadata())?,
    })
}

/// The model line appended to the native system prompt: the configured served
/// model ID and, when recognized, its family name.
pub fn model_identity(model: &str) -> String {
    let family = if model.starts_with("qwen3.8-flash-next-") { "Qwen3.8 Flash Next" } else { "not specified" };
    format!("Served model id (configured): {model:?}; family: {family}.")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn memory() -> Session {
        Session {
            history: Vec::new(),
            bytes: 0,
            metadata: Metadata::default(),
            pruned_turns: 0,
            store: None,
        }
    }
    #[test]
    fn preserves_exact_contents_and_complete_pairs() {
        let mut s = memory();
        s.complete("  user\n".into(), "answer\t".into()).unwrap();
        assert_eq!(s.history()[0].content, "  user\n");
        assert_eq!(s.history()[1].content, "answer\t");
        for i in 0..MAX_HISTORY_TURNS {
            s.complete(i.to_string(), "a".into()).unwrap();
        }
        assert_eq!(s.turns(), MAX_HISTORY_TURNS);
        assert_eq!(s.pruned_turns, 1);
        assert_eq!(s.history()[0].role, "user");
        assert!(s.undo());
        assert_eq!(s.turns(), MAX_HISTORY_TURNS - 1);
        s.clear();
        assert_eq!(s.history_bytes(), 0);
        assert!(!s.undo());
    }
    #[test]
    fn byte_budget_never_splits_a_turn() {
        let mut s = memory();
        s.complete("first".into(), "answer".into()).unwrap();
        assert!(
            s.complete("x".repeat(MAX_HISTORY_BYTES), String::new())
                .unwrap()
        );
        assert_eq!(s.turns(), 1);
        assert_eq!(s.pruned_turns, 1);
        assert!(
            !s.complete("too large".into(), "x".repeat(MAX_HISTORY_BYTES))
                .unwrap()
        );
        assert_eq!(s.turns(), 1);
        assert_eq!(s.metadata.completed_turns, 3);
    }
    #[test]
    fn metadata_format_cannot_contain_conversation() {
        assert_eq!(
            parse_metadata("vade-session-v1\ncompleted_turns=42\n")
                .unwrap()
                .completed_turns,
            42
        );
        for input in [
            "",
            "vade-session-v1\ncompleted_turns=1\nsecret",
            "vade-session-v1\ncompleted_turns=-1\n",
            "vade-session-v1\ncompleted_turns=18446744073709551616\n",
        ] {
            assert!(parse_metadata(input).is_err());
        }
    }
    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn persists_only_private_metadata_and_refuses_redirects() {
        use std::fs;
        use std::os::unix::fs::{PermissionsExt, symlink};
        let path = std::env::temp_dir().join(format!("vade-session-test-{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        let workspace = Workspace::new(&path).unwrap();
        let mut session = Session::new(&workspace, Some("example")).unwrap();
        session
            .complete(
                "private prompt marker".into(),
                "private output marker".into(),
            )
            .unwrap();
        let file = path.join(".vade/sessions/example.state");
        assert_eq!(
            fs::read_to_string(&file).unwrap(),
            "vade-session-v1\ncompleted_turns=1\n"
        );
        assert_eq!(
            fs::metadata(&file).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(Session::new(&workspace, Some("example")).is_err());
        drop(session);
        let resumed = Session::new(&workspace, Some("example")).unwrap();
        assert_eq!(resumed.metadata.completed_turns, 1);
        assert_eq!(resumed.turns(), 0);
        symlink(&file, path.join(".vade/sessions/redirect.state")).unwrap();
        assert!(Session::new(&workspace, Some("redirect")).is_err());
        assert!(Session::new(&workspace, Some("../escape")).is_err());
        fs::set_permissions(path.join(".vade"), fs::Permissions::from_mode(0o755)).unwrap();
        assert!(Session::new(&workspace, Some("example")).is_err());
        fs::remove_dir_all(path).unwrap();
    }
    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn session_lock_release_is_independent_of_duplicate_descriptor_lifetime() {
        use std::fs;
        struct Fixture(std::path::PathBuf);
        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let fixture = Fixture(std::env::temp_dir().join(format!(
            "vade-session-duplicate-test-{}",
            std::process::id()
        )));
        fs::create_dir(&fixture.0).unwrap();
        let workspace = Workspace::new(&fixture.0).unwrap();
        let mut session = Session::new(&workspace, Some("example")).unwrap();
        session.complete("first".into(), "answer".into()).unwrap();
        // dup and a child before exec share the same open file description.
        // Holding this descriptor makes the old close-only race deterministic.
        let inherited = session
            .store
            .as_ref()
            .unwrap()
            ._lock
            .duplicate_descriptor()
            .unwrap();
        assert!(Session::new(&workspace, Some("example")).is_err());
        drop(session);
        let mut resumed = Session::new(&workspace, Some("example"))
            .expect("owner release must unlock even while a duplicate is still open");
        assert_eq!(resumed.metadata.completed_turns, 1);
        assert!(Session::new(&workspace, Some("example")).is_err());
        drop(inherited);
        assert!(
            Session::new(&workspace, Some("example")).is_err(),
            "closing the old duplicate must not release the new owner's lock"
        );
        resumed.complete("second".into(), "answer".into()).unwrap();
        drop(resumed);
        let final_session = Session::new(&workspace, Some("example")).unwrap();
        assert_eq!(final_session.metadata.completed_turns, 2);
    }
}
