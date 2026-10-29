//! Private, locked, append-only native sessions. A record is committed only when
//! its length trailer and JSON are complete; a torn final record is truncated on
//! explicit resume. Tool intent is synced before execution; never auto-replayed.
use crate::agent::{Conversation, Limits, RunObserver};
use crate::agent_wire::{FinishReason, Message};
use crate::json::{self, Value};
use crate::tools::Workspace;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};

const MAX_JOURNAL: u64 = 64 * 1024 * 1024;
const MAX_RECORDS: u64 = 8192;
/// A native session's private, locked, append-only record of every committed
/// message, compaction and checkpoint, used to resume or fork it exactly.
pub struct Journal {
    name: String,
    file: File,
    _lock: crate::tools::anchored::Lock,
    seq: u64,
    len: usize,
    bytes: u64,
    poisoned: bool,
    last_tail: Option<(usize, String)>,
    artifacts: crate::artifacts::Archive,
    records: Vec<(u64, usize)>,
    generation: u64,
    last_compaction: Option<(u64, Value)>,
    workspace_id: String,
    parent: Option<Value>,
    checkpoint: Option<crate::branch::Checkpoint>,
    kv_hints: Vec<crate::kv::Hint>,
    reply_cutoffs: usize,
    generation_seed: Option<u64>,
}
fn error(s: &str) -> io::Error {
    io::Error::other(s)
}
/// True for a usable session name: 1 to 48 ASCII letters, digits, `-` or `_`.
pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 48
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}
/// A new session name from the time, the process ID and a counter.
pub fn fresh_name() -> String {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    format!(
        "session-{}-{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}
/// The names of the sessions saved in this workspace, at most 256.
pub fn list(workspace: &Workspace) -> io::Result<Vec<String>> {
    let entries = workspace.session_dir()?.names(256)?;
    let limited = entries.iter().any(|s| s == "[listing limit reached]");
    let mut names: Vec<_> = entries
        .into_iter()
        .filter_map(|s| {
            s.strip_suffix(".journal")
                .filter(|n| valid_name(n))
                .map(str::to_owned)
        })
        .collect();
    names.sort();
    if limited {
        names.push("[session listing limit reached]".into());
    }
    Ok(names)
}
impl Journal {
    fn open(workspace: &Workspace, name: &str) -> io::Result<Self> {
        if !valid_name(name) {
            return Err(error(
                "session name needs 1–48 ASCII letters, digits, '-' or '_'",
            ));
        }
        let dir = workspace.session_dir()?;
        let lock = dir.lock(&format!("{name}.journal-lock"))?;
        let file = dir.journal_file(&format!("{name}.journal"))?;
        Self::from_locked_file(name, file, lock, crate::branch::workspace_id(workspace)?)
    }
    fn from_locked_file(
        name: &str,
        file: File,
        lock: crate::tools::anchored::Lock,
        workspace_id: String,
    ) -> io::Result<Self> {
        if file.metadata()?.len() > MAX_JOURNAL {
            return Err(error("session journal exceeds 64 MiB"));
        }
        Ok(Self {
            name: name.into(),
            file,
            _lock: lock,
            seq: 0,
            len: 0,
            bytes: 0,
            poisoned: false,
            last_tail: None,
            artifacts: crate::artifacts::Archive::default(),
            records: Vec::new(),
            generation: 0,
            last_compaction: None,
            workspace_id,
            parent: None,
            checkpoint: None,
            kv_hints: Vec::new(),
            reply_cutoffs: 0,
            generation_seed: None,
        })
    }
    pub fn create(
        workspace: &Workspace,
        name: &str,
        conversation: &Conversation,
    ) -> io::Result<Self> {
        let mut s = Self::open(workspace, name)?;
        if s.file.metadata()?.len() != 0 {
            return Err(error(
                "session exists; explicitly resume it or choose another name",
            ));
        }
        s.append_record(
            0,
            conversation.messages(),
            Some(
                conversation
                    .request(crate::agent_wire::WireLimits::default())
                    .serialized_options(),
            ),
            Some(("workspace", Value::String(s.workspace_id.clone()))),
        )?;
        s.save_hints(&conversation.kv_hints)?;
        Ok(s)
    }
    pub fn resume(
        workspace: &Workspace,
        name: &str,
        limits: Limits,
    ) -> io::Result<(Self, Conversation, bool)> {
        // Do not create a typo'd session on a read operation.
        if !valid_name(name)
            || workspace
                .session_dir()?
                .optional_file(&format!("{name}.journal"))?
                .is_none()
        {
            return Err(error("session not found"));
        }
        Self::replay(Self::open(workspace, name)?, limits)
    }
    fn replay(mut s: Self, limits: Limits) -> io::Result<(Self, Conversation, bool)> {
        let file_len = s.file.metadata()?.len();
        let mut options = None;
        let mut messages: Vec<Value> = Vec::new();
        let mut sizes: Vec<usize> = Vec::new();
        let mut history_bytes = 0usize;
        let mut torn = false;
        while s.bytes < file_len {
            if s.seq >= MAX_RECORDS {
                return Err(error("session record count exceeds limit"));
            }
            let mut header = [0u8; 8];
            if file_len - s.bytes < 8 {
                torn = true;
                break;
            }
            s.file.read_exact(&mut header)?;
            let len = u64::from_le_bytes(header);
            if len > json::MAX_INPUT_BYTES as u64 {
                return Err(error("invalid journal record length"));
            }
            if file_len - s.bytes - 8 < len + 8 {
                torn = true;
                break;
            }
            let mut raw = vec![0; len as usize];
            s.file.read_exact(&mut raw)?;
            let mut trailer = [0u8; 8];
            s.file.read_exact(&mut trailer)?;
            if trailer != header {
                return Err(error("journal record trailer corrupt"));
            }
            let record = json::parse_bytes(&raw).map_err(|_| error("journal record corrupt"))?;
            let number = |key| -> io::Result<u64> {
                if let Some(Value::Number(n)) = record.get(key) {
                    n.parse().map_err(|_| error("invalid journal number"))
                } else {
                    Err(error("missing journal number"))
                }
            };
            if number("v")? != 1 || number("seq")? != s.seq {
                return Err(error("journal version/sequence mismatch"));
            }
            let keep =
                usize::try_from(number("keep")?).map_err(|_| error("invalid journal keep"))?;
            if keep > messages.len() {
                return Err(error("journal references missing history"));
            }
            if let Some(identity) = record.get("workspace") {
                if s.seq != 0 || identity.as_str() != Some(&s.workspace_id) {
                    return Err(error("journal belongs to a different workspace directory"));
                }
            }
            if s.seq == 0 {
                options = Some(
                    record
                        .get("options")
                        .cloned()
                        .ok_or_else(|| error("journal has no options"))?,
                );
            } else if record.get("options").is_some() {
                return Err(error("journal options changed"));
            }
            if record.get("generation_seed").is_some() {
                s.generation_seed = Some(number("generation_seed")?);
            }
            let append = record
                .get("messages")
                .and_then(Value::as_array)
                .ok_or_else(|| error("journal has no messages"))?;
            // Include legacy journal markers in the local notice, while
            // Message ingestion removes them from all replayed wire messages.
            if record.get("reply_cutoff") == Some(&Value::Bool(true))
                || append
                    .iter()
                    .any(|m| m.get("vade_truncated") == Some(&Value::Bool(true)))
            {
                s.reply_cutoffs += 1;
            }
            if let Some(metadata) = record.get("compaction") {
                let n = |key: &str| -> io::Result<u64> {
                    match metadata.get(key) {
                        Some(Value::Number(n)) => {
                            n.parse().map_err(|_| error("invalid compaction metadata"))
                        }
                        _ => Err(error("missing compaction metadata")),
                    }
                };
                if s.seq == 0
                    || keep != 0
                    || n("generation")? != s.generation + 1
                    || n("source_record")? != s.seq - 1
                    || n("source_messages")? != messages.len() as u64
                {
                    return Err(error("compaction source/generation mismatch"));
                }
                s.generation += 1;
                s.last_compaction = Some((s.seq, metadata.clone()));
            }
            if let Some(value) = record.get("checkpoint") {
                let checkpoint = crate::branch::Checkpoint::from_value(value)?;
                if !checkpoint.belongs_to(&s.workspace_id)
                    || keep != messages.len()
                    || !append.is_empty()
                {
                    return Err(error("checkpoint workspace or history mismatch"));
                }
                s.checkpoint = Some(checkpoint);
            }
            if let Some(parent) = record.get("branch") {
                let name = parent
                    .get("parent_session")
                    .and_then(Value::as_str)
                    .ok_or_else(|| error("branch parent missing"))?;
                let number = |key| match parent.get(key) {
                    Some(Value::Number(n)) => {
                        n.parse::<u64>().map_err(|_| error("invalid branch number"))
                    }
                    _ => Err(error("missing branch number")),
                };
                let label = parent
                    .get("task_label")
                    .and_then(Value::as_str)
                    .ok_or_else(|| error("branch label missing"))?;
                if !valid_name(name)
                    || s.seq == 0
                    || number("parent_record")? != s.seq - 1
                    || number("context_generation")? != s.generation
                    || label.len() > 1024
                    || keep != messages.len()
                    || !append.is_empty()
                    || s.checkpoint.is_none()
                    || parent.get("engine_checkpoint") != Some(&Value::Null)
                {
                    return Err(error("invalid branch checkpoint"));
                }
                s.parent = Some(parent.clone());
            }
            s.kv_hints.retain(|h| h.messages <= keep);
            if let Some(value) = record.get("engine_checkpoints") {
                s.kv_hints = crate::kv::parse_hints(value, keep + append.len());
            }
            history_bytes -= sizes.drain(keep..).sum::<usize>();
            messages.truncate(keep);
            for message in append {
                let bytes = json::stringify(message)?.len();
                history_bytes = history_bytes.saturating_add(bytes);
                if messages.len() >= 32768 || history_bytes > limits.max_history_bytes {
                    return Err(error("journal history exceeds limit"));
                }
                sizes.push(bytes);
                messages.push(message.clone());
            }
            if let Some(artifact) = record.get("artifact") {
                let id = format!("a-{}", s.seq);
                let created = match artifact.get("created") {
                    Some(Value::Number(n)) => n
                        .parse::<u64>()
                        .map_err(|_| error("invalid artifact time"))?,
                    _ => return Err(error("missing artifact time")),
                };
                let text = artifact
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| error("missing artifact content"))?;
                s.artifacts.insert(id, created, text.into())?;
            }
            s.records.push((s.bytes + 8, len as usize));
            s.seq += 1;
            s.bytes += len + 16;
        }
        let Some(Value::Object(mut fields)) = options else {
            return Err(error("session has no complete initial record"));
        };
        fields.push(("messages".into(), Value::Array(messages)));
        let mut conversation = Conversation::from_request(Value::Object(fields), limits)?;
        conversation.kv_hints = s.kv_hints.clone();
        s.len = conversation.messages().len();
        if torn {
            s.file.set_len(s.bytes)?;
            s.file.sync_all()?;
        }
        s.file.seek(SeekFrom::End(0))?;
        Ok((s, conversation, torn))
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn poisoned(&self) -> bool {
        self.poisoned
    }
    pub fn generation_seed(&self) -> Option<u64> {
        self.generation_seed
    }
    pub fn set_generation_seed(&mut self, seed: u64) -> io::Result<()> {
        if self.generation_seed != Some(seed) {
            self.append_record(
                self.len,
                &[],
                None,
                Some(("generation_seed", Value::Number(seed.to_string()))),
            )?;
            self.generation_seed = Some(seed);
        }
        Ok(())
    }
    /// Explicit answer regeneration changes only the last prose message in the
    /// logical view. Verify the source; keep completed tools and raw records.
    pub fn retry_answer(
        &mut self,
        c: &Conversation,
        seed: u64,
        limits: Limits,
    ) -> io::Result<Conversation> {
        self.retry_answer_with_optional_seed(c, Some(seed), limits)
    }
    /// Keep an unspecified sampler unspecified when retrying a fresh default
    /// session. A recorded seed cannot be silently removed from a saved session.
    pub fn retry_answer_with_optional_seed(
        &mut self,
        c: &Conversation,
        seed: Option<u64>,
        limits: Limits,
    ) -> io::Result<Conversation> {
        if seed.is_none() && self.generation_seed.is_some() {
            return Err(error("answer retry cannot discard a recorded sampling seed"));
        }
        let source = self.view_at(self.records.len().saturating_sub(1), limits)?;
        if c.messages().len() != self.len
            || source.request(limits.wire).to_json()? != c.request(limits.wire).to_json()?
        {
            return Err(error("answer retry source no longer matches the journal"));
        }
        let replacement = c.without_last_answer(limits)?;
        self.save_hints(&[])?;
        self.append_record(
            replacement.messages().len(),
            &[],
            None,
            seed.map(|seed| ("generation_seed", Value::Number(seed.to_string()))),
        )?;
        self.generation_seed = seed;
        self.kv_hints.clear();
        Ok(replacement)
    }
    pub fn cutoff_notice(&self) -> Option<String> {
        (self.reply_cutoffs > 0).then(|| {
            format!(
                "Journal records {} reply cutoff(s) at an upstream length limit. Truncated text was saved without incomplete tool calls; /journal inspects the original records. Resuming does not replay tools or automatically continue an earlier cutoff.",
                self.reply_cutoffs
            )
        })
    }
    pub fn save(&mut self, c: &Conversation) -> io::Result<()> {
        if self.len > c.messages().len() {
            return Err(error("journal/history mismatch"));
        }
        if self.len < c.messages().len() {
            self.append(self.len, &c.messages()[self.len..], None)?;
        }
        self.save_hints(&c.kv_hints)
    }
    fn save_hints(&mut self, hints: &[crate::kv::Hint]) -> io::Result<()> {
        if self.kv_hints != hints {
            self.append_record(
                self.len,
                &[],
                None,
                Some(("engine_checkpoints", crate::kv::hints_value(hints))),
            )?;
            self.kv_hints = hints.to_vec();
        }
        Ok(())
    }
    /// Deletes this locked journal only. The owner passes the returned private
    /// hints to NativeModel::release_branch after deletion, off the UI thread.
    /// Lock files stay in place so another process cannot acquire a second lock.
    pub fn delete(self, workspace: &Workspace) -> io::Result<Vec<crate::kv::Hint>> {
        use std::os::unix::fs::MetadataExt;
        if crate::branch::workspace_id(workspace)? != self.workspace_id {
            return Err(error("session workspace differs from selected workspace"));
        }
        let dir = workspace.session_dir()?;
        let filename = format!("{}.journal", self.name);
        let file = dir
            .optional_file(&filename)?
            .ok_or_else(|| error("session missing"))?;
        let current = file.metadata()?;
        let owned = self.file.metadata()?;
        if current.dev() != owned.dev() || current.ino() != owned.ino() || current.nlink() != 1 {
            return Err(error("session changed before deletion"));
        }
        workspace.verify_root()?;
        dir.remove(&filename)?;
        Ok(self.kv_hints.clone())
    }
    pub fn compact(&mut self, plan: &crate::context::CompactionPlan) -> io::Result<()> {
        if plan.source_messages != self.len || self.seq == 0 {
            return Err(error("compaction plan does not match journal state"));
        }
        let source = self.view_at(self.records.len() - 1, plan.limits)?;
        if source.request(plan.limits.wire).to_json()? != plan.source_request {
            return Err(error(
                "compaction plan is stale; preview the current journal view again",
            ));
        }
        let metadata = plan.metadata(self.generation + 1, self.seq - 1);
        let record = self.seq;
        self.append_record(
            0,
            plan.conversation().messages(),
            None,
            Some(("compaction", metadata.clone())),
        )?;
        self.kv_hints.clear();
        self.generation += 1;
        self.last_compaction = Some((record, metadata));
        Ok(())
    }
    /// Host diagnostic only; never inject a failed plan into model history.
    pub fn record_compaction_failure(&mut self, reason: &str) -> io::Result<()> {
        if reason.is_empty() || reason.len() > 1024 || reason.chars().any(char::is_control) {
            return Err(error("invalid compaction failure reason"));
        }
        self.append_record(
            self.len,
            &[],
            None,
            Some(("compaction_failure", Value::String(reason.into()))),
        )
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn record_count(&self) -> usize {
        self.records.len()
    }
    pub fn compaction_status(&self) -> String {
        let Some((record, m)) = &self.last_compaction else {
            return format!(
                "Context generation {}; no recorded compaction in this journal. Engine cache/reuse unknown.",
                self.generation
            );
        };
        let field = |key| {
            m.get(key)
                .and_then(|v| match v {
                    Value::Number(n) => Some(n.as_str()),
                    _ => None,
                })
                .unwrap_or("unknown")
        };
        format!(
            "Context generation {} · compaction record {} · source records 0..={} · {} → {} request bytes · unchanged prefix {} messages.\nExact input tokens [before, after]: {} (null = not counted).\n{}\n{} summary:\n{}\nInspect source with /journal RECORD OFFSET LENGTH (UTF-8 bytes).",
            self.generation,
            record,
            field("source_record"),
            field("before_bytes"),
            field("after_bytes"),
            field("unchanged_prefix_messages"),
            m.get("exact_input_tokens")
                .and_then(|v| crate::json::stringify(v).ok())
                .unwrap_or_else(|| "null".into()),
            crate::context::KV_COMPACTION_NOTICE,
            match m.get("summary_origin").and_then(Value::as_str) {
                Some("local excerpts") => "Local excerpt".into(),
                Some("model") => format!("Model-written (kept {} turns, {} tokens)", field("kept_turns"), field("summary_tokens")),
                _ => "User-supplied".into(),
            },
            m.get("summary")
                .and_then(Value::as_str)
                .unwrap_or("unavailable")
        )
    }
    pub fn inspect_record(
        &self,
        record: usize,
        offset: usize,
        length: usize,
    ) -> io::Result<String> {
        use std::os::unix::fs::FileExt;
        let &(position, bytes) = self
            .records
            .get(record)
            .ok_or_else(|| error("journal record not found"))?;
        let mut data = vec![0; bytes];
        // pread leaves the append cursor untouched. The held journal lock also
        // serializes against the worker, so inspection cannot corrupt writes.
        self.file.read_exact_at(&mut data, position)?;
        let text = std::str::from_utf8(&data).map_err(|_| error("journal record is not UTF-8"))?;
        // Keep byte offsets stable while preventing capabilities from reaching
        // terminal logs, including range inspections that split a handle.
        let mut redacted = data.clone();
        if let Some(hints) = json::view(text)?
            .get("engine_checkpoints")
            .and_then(json::View::array)
        {
            for hint in hints {
                if let Some(handle) = hint.get("checkpoint").and_then(|v| v.get("handle")) {
                    let raw = handle.raw();
                    let start = raw.as_ptr() as usize - text.as_ptr() as usize;
                    let quoted = usize::from(raw.starts_with('"') && raw.ends_with('"'));
                    redacted[start + quoted..start + raw.len() - quoted].fill(b'*');
                }
            }
        }
        let redacted =
            std::str::from_utf8(&redacted).map_err(|_| error("journal redaction encoding"))?;
        crate::artifacts::chunk(
            redacted,
            &format!("journal-record-{record}"),
            offset,
            length,
            8192,
        )
    }
    /// Reconstruct a bounded historical model view; never execute a tool. The
    /// source record is a durable checkpoint, not a provider cache handle.
    pub fn view_at(&self, record: usize, limits: Limits) -> io::Result<Conversation> {
        if record >= self.records.len() {
            return Err(error("journal record not found"));
        }
        let mut options = None;
        let mut messages = Vec::new();
        let mut sizes = Vec::new();
        let mut bytes = 0usize;
        let mut hints: Vec<crate::kv::Hint> = Vec::new();
        for i in 0..=record {
            let value = json::parse_bytes(&self.record_bytes(i)?)?;
            let keep = match value.get("keep") {
                Some(Value::Number(n)) => n
                    .parse::<usize>()
                    .map_err(|_| error("invalid journal keep"))?,
                _ => return Err(error("missing journal keep")),
            };
            if keep > messages.len() {
                return Err(error("journal references missing history"));
            }
            if i == 0 {
                options = value.get("options").cloned();
            }
            hints.retain(|h| h.messages <= keep);
            bytes -= sizes.drain(keep..).sum::<usize>();
            messages.truncate(keep);
            for message in value
                .get("messages")
                .and_then(Value::as_array)
                .ok_or_else(|| error("missing journal messages"))?
            {
                let length = json::stringify(message)?.len();
                bytes += length;
                if bytes > limits.max_history_bytes || messages.len() >= 32768 {
                    return Err(error("historical view exceeds limits"));
                }
                sizes.push(length);
                messages.push(message.clone());
            }
            if let Some(value) = value.get("engine_checkpoints") {
                hints = crate::kv::parse_hints(value, messages.len());
            }
        }
        let Some(Value::Object(mut fields)) = options else {
            return Err(error("missing journal options"));
        };
        fields.push(("messages".into(), Value::Array(messages)));
        let mut c = Conversation::from_request(Value::Object(fields), limits)?;
        c.kv_hints = hints;
        Ok(c)
    }
    fn record_bytes(&self, record: usize) -> io::Result<Vec<u8>> {
        use std::os::unix::fs::FileExt;
        let &(position, bytes) = self
            .records
            .get(record)
            .ok_or_else(|| error("journal record not found"))?;
        let mut data = vec![0; bytes];
        self.file.read_exact_at(&mut data, position)?;
        Ok(data)
    }
    pub fn branch_status(&self, workspace: &Workspace) -> io::Result<String> {
        if crate::branch::workspace_id(workspace)? != self.workspace_id {
            return Err(error("session workspace differs from selected workspace"));
        }
        let origin = if let Some(parent) = &self.parent {
            let name = parent
                .get("parent_session")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let record = parent
                .get("parent_record")
                .and_then(|v| match v {
                    Value::Number(n) => Some(n.as_str()),
                    _ => None,
                })
                .unwrap_or("unknown");
            let label = parent
                .get("task_label")
                .and_then(Value::as_str)
                .unwrap_or("");
            format!(
                "Branch {} · parent {} at record {} · context generation {} · task label: {}.\n",
                self.name, name, record, self.generation, label
            )
        } else {
            format!(
                "Session {} · root transcript · context generation {}.\n",
                self.name, self.generation
            )
        };
        let state = match &self.checkpoint {
            Some(checkpoint) => checkpoint.inspect(workspace)?,
            None => "No file revision checkpoint. Re-read relevant files before acting on old facts. Switching never undoes edits. KV hints are optional; KV reuse unknown until a subsequent Engine response reports it.".into(),
        };
        let valid = self
            .kv_hints
            .iter()
            .filter(|h| h.checkpoint.expires_at_ms > crate::kv::now_ms())
            .count();
        Ok(format!(
            "{origin}{state}\nEngine checkpoint hints: {valid} unexpired / {} recorded (canonical inputs, not generated tails). Full transcript replay remains authoritative.",
            self.kv_hints.len()
        ))
    }
    /// Copy the complete bounded raw journal so inherited artifacts retain IDs,
    /// expiry and provenance. Atomic exclusive installation avoids a half-fork.
    pub fn fork(
        &mut self,
        workspace: &Workspace,
        name: &str,
        label: &str,
        limits: Limits,
    ) -> io::Result<(Self, Conversation)> {
        use std::os::unix::fs::FileExt;
        if self.poisoned
            || !valid_name(name)
            || name == self.name
            || label.len() > 1024
            || crate::terminal::sanitize(label) != label
            || self.seq + 2 >= MAX_RECORDS
        {
            return Err(error(
                "cannot fork: unavailable journal, invalid name/label, or journal capacity",
            ));
        }
        if crate::branch::workspace_id(workspace)? != self.workspace_id {
            return Err(error("fork workspace differs from source"));
        }
        let c = self.view_at(self.records.len().saturating_sub(1), limits)?;
        if crate::context::has_unknown_outcome(&c) {
            return Err(error(
                "inspect unknown tool outcomes before forking; tools are never replayed",
            ));
        }
        let dir = workspace.session_dir()?;
        let lock = dir.lock(&format!("{name}.journal-lock"))?;
        let filename = format!("{name}.journal");
        if dir.optional_file(&filename)?.is_some() {
            return Err(error("fork destination exists; choose a new name"));
        }
        let checkpoint = crate::branch::Checkpoint::capture(workspace, &c)?;
        self.append_record(
            self.len,
            &[],
            None,
            Some(("checkpoint", checkpoint.value())),
        )?;
        self.checkpoint = Some(checkpoint);
        let parent = Value::Object(vec![
            ("parent_session".into(), Value::String(self.name.clone())),
            (
                "parent_record".into(),
                Value::Number((self.seq - 1).to_string()),
            ),
            (
                "context_generation".into(),
                Value::Number(self.generation.to_string()),
            ),
            ("task_label".into(), Value::String(label.into())),
            (
                "active_user_message".into(),
                c.messages()
                    .iter()
                    .rposition(|m| m.role() == "user")
                    .map_or(Value::Null, |i| Value::Number(i.to_string())),
            ),
            ("engine_checkpoint".into(), Value::Null),
        ]);
        let record = format!(
            "{{\"v\":1,\"seq\":{},\"keep\":{},\"messages\":[],\"branch\":{}}}",
            self.seq,
            self.len,
            json::stringify(&parent)?
        );
        if self.bytes + record.len() as u64 + 16 > MAX_JOURNAL {
            return Err(error("fork would exceed journal disk limit"));
        }
        let mut raw = Vec::with_capacity(self.bytes as usize + record.len() + 16);
        raw.resize(self.bytes as usize, 0);
        self.file.read_exact_at(&mut raw, 0)?;
        let length = (record.len() as u64).to_le_bytes();
        raw.extend_from_slice(&length);
        raw.extend_from_slice(record.as_bytes());
        raw.extend_from_slice(&length);
        dir.install(&filename, &raw, 0o600, true, || workspace.verify_root())?;
        let child = Self::from_locked_file(
            name,
            dir.journal_file(&filename)?,
            lock,
            self.workspace_id.clone(),
        )?;
        let (child, conversation, torn) = Self::replay(child, limits)?;
        if torn {
            return Err(error("unexpected incomplete fork"));
        }
        Ok((child, conversation))
    }
    fn append(
        &mut self,
        keep: usize,
        messages: &[Message],
        options: Option<&str>,
    ) -> io::Result<()> {
        self.append_record(keep, messages, options, None)
    }
    fn checkpoint_with_metadata(
        &mut self,
        base: &Conversation,
        staged: &[Message],
        metadata: Option<(&str, Value)>,
    ) -> io::Result<()> {
        // A staged tail can already be in the log. Each revision replaces it in
        // the logical view while prior durable records remain append-only.
        if base.messages().len() > self.len {
            self.save(base)?;
        }
        self.save_hints(&base.kv_hints)?;
        self.append_record(base.messages().len(), staged, None, metadata)
    }
    fn append_record(
        &mut self,
        keep: usize,
        messages: &[Message],
        options: Option<&str>,
        extra: Option<(&str, Value)>,
    ) -> io::Result<()> {
        if self.poisoned {
            return Err(error(
                "journal is unavailable after a write failure; restart and resume",
            ));
        }
        let mut payload = String::from("[");
        for (index, message) in messages.iter().enumerate() {
            if index > 0 {
                payload.push(',');
            }
            payload.push_str(message.serialized_json());
        }
        payload.push(']');
        if options.is_none()
            && extra.is_none()
            && self
                .last_tail
                .as_ref()
                .is_some_and(|(k, p)| *k == keep && *p == payload)
        {
            return Ok(());
        }
        let mut record = format!(
            "{{\"v\":1,\"seq\":{},\"keep\":{},\"messages\":{}",
            self.seq, keep, payload
        );
        if let Some(options) = options {
            record.push_str(",\"options\":");
            record.push_str(options);
        }
        let reply_cutoff = extra
            .as_ref()
            .is_some_and(|(key, value)| *key == "reply_cutoff" && *value == Value::Bool(true));
        if let Some((key, value)) = extra {
            record.push(',');
            record.push_str(&json::quote(key));
            record.push(':');
            record.push_str(&json::stringify(&value)?);
        }
        record.push('}');
        if record.len() > json::MAX_INPUT_BYTES
            || self.bytes + record.len() as u64 + 16 > MAX_JOURNAL
            || self.seq >= MAX_RECORDS
        {
            self.poisoned = true;
            return Err(error("journal capacity reached; start a new session"));
        }
        let length = (record.len() as u64).to_le_bytes();
        let result = (|| {
            self.file.write_all(&length)?;
            self.file.write_all(record.as_bytes())?;
            self.file.write_all(&length)?;
            self.file.sync_all()
        })();
        if result.is_err() {
            self.poisoned = true;
            return Err(error("journal sync failed; stop and resume after restart"));
        }
        self.records.push((self.bytes + 8, record.len()));
        self.bytes += record.len() as u64 + 16;
        self.seq += 1;
        self.len = keep + messages.len();
        self.reply_cutoffs += usize::from(reply_cutoff);
        self.last_tail = Some((keep, payload));
        Ok(())
    }
}
impl crate::artifacts::Access for Journal {
    fn check_capacity(&mut self) -> io::Result<()> {
        if self.poisoned
            || self.bytes + 8 * crate::artifacts::MAX_ARTIFACT_BYTES as u64 > MAX_JOURNAL
            || self.seq + 3 >= MAX_RECORDS
        {
            return Err(error("journal capacity unavailable for tool output"));
        }
        self.artifacts.check_capacity()
    }
    fn retain(&mut self, text: &str) -> io::Result<String> {
        self.check_capacity()?;
        if text.len() > crate::artifacts::MAX_ARTIFACT_BYTES {
            return Err(error("artifact capture too large"));
        }
        let created = crate::artifacts::now();
        let id = format!("a-{}", self.seq);
        self.append_record(
            self.len,
            &[],
            None,
            Some((
                "artifact",
                Value::Object(vec![
                    ("created".into(), Value::Number(created.to_string())),
                    ("text".into(), Value::String(text.into())),
                ]),
            )),
        )?;
        self.artifacts.insert(id.clone(), created, text.into())?;
        Ok(id)
    }
    fn read(&mut self, id: &str, offset: usize, length: usize, cap: usize) -> io::Result<String> {
        self.artifacts.read(id, offset, length, cap)
    }
}
impl RunObserver for Journal {
    fn checkpoint(&mut self, base: &Conversation, staged: &[Message]) -> io::Result<()> {
        self.checkpoint_with_metadata(base, staged, None)
    }
    fn checkpoint_reply(
        &mut self,
        base: &Conversation,
        message: &Message,
        finish: &FinishReason,
    ) -> io::Result<()> {
        let metadata =
            (*finish == FinishReason::Length).then_some(("reply_cutoff", Value::Bool(true)));
        self.checkpoint_with_metadata(base, std::slice::from_ref(message), metadata)
    }
}

#[cfg(test)]
mod tests;
