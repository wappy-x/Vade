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
pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 48
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}
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
            if m.get("summary_origin").and_then(Value::as_str) == Some("local excerpts") {
                "Local excerpt"
            } else {
                "User-supplied"
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
mod tests {
    use super::*;
    use crate::agent::{self, Model, ModelRequest, ToolError, ToolExecutor, ToolInvocation};
    use crate::agent_wire::AssistantTurn;
    use crate::test_workspace::Fixture;
    use std::sync::atomic::AtomicBool;
    fn conversation() -> Conversation {
        Conversation::from_request(json::parse(r#"{"model":"mock","tools":[{"type":"function","function":{"name":"mock_write"}}],"messages":[{"role":"system","content":"local mock"}]}"#).unwrap(), Limits::default()).unwrap()
    }
    fn user(c: &mut Conversation, text: &str) {
        c.push_user(
            Value::Object(vec![
                ("role".into(), Value::String("user".into())),
                ("content".into(), Value::String(text.into())),
            ]),
            Limits::default(),
        )
        .unwrap();
    }
    #[test]
    fn default_retry_omits_seed_metadata_and_cannot_discard_a_recorded_seed() {
        for seed in [None, Some(713)] {
            let fixture = Fixture::new();
            let workspace = fixture.workspace();
            let conversation = Conversation::from_request(
                json::parse(r#"{"model":"mock","messages":[{"role":"system","content":"rules"},{"role":"user","content":"question"},{"role":"assistant","content":"answer"}]}"#).unwrap(),
                Limits::default(),
            ).unwrap();
            let mut journal = Journal::create(&workspace, "retry-policy", &conversation).unwrap();
            let expected = if let Some(seed) = seed {
                journal.set_generation_seed(seed).unwrap();
                let count = journal.record_count();
                assert!(journal.retry_answer_with_optional_seed(&conversation, None, Limits::default()).is_err());
                assert_eq!(journal.record_count(), count, "a refused default retry must not mutate the journal");
                journal.retry_answer(&conversation, seed, Limits::default()).unwrap();
                Some(seed)
            } else {
                journal.retry_answer_with_optional_seed(&conversation, None, Limits::default()).unwrap();
                None
            };
            drop(journal);
            let (restored, conversation, torn) = Journal::resume(&workspace, "retry-policy", Limits::default()).unwrap();
            assert!(!torn);
            assert_eq!(restored.generation_seed(), expected);
            assert_eq!(conversation.messages().len(), 2);
        }
    }
    #[test]
    fn optional_kv_hints_survive_forks_but_not_compaction_or_invalid_metadata() {
        let fixture = Fixture::new();
        let w = fixture.workspace();
        let mut c = Conversation::from_request(json::parse(&format!(r#"{{"model":"mock","messages":[{{"role":"system","content":"rules"}},{{"role":"user","content":"old"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"current"}},{{"role":"assistant","content":"answer"}}]}}"#, "resolved detail ".repeat(200))).unwrap(), Limits::default()).unwrap();
        let hint = crate::kv::Hint {
            checkpoint: crate::kv::Checkpoint {
                handle: "b".repeat(64),
                cache_domain: "scope".into(),
                prompt_tokens: 100,
                reusable_prefix_tokens: 96,
                expires_at_ms: crate::kv::now_ms() + 10000,
                retained_bytes: 1000,
                scope_retained_bytes: 1000,
            },
            messages: 2,
        };
        c.kv_hints.push(hint.clone());
        let mut journal = Journal::create(&w, "parent-kv", &c).unwrap();
        let (mut child, fork) = journal
            .fork(&w, "child-kv", "task", Limits::default())
            .unwrap();
        assert_eq!(fork.checkpoint_hints(), &[hint.clone()]);
        let plan = crate::context::plan(&c, "Old detail resolved.", 1, Limits::default()).unwrap();
        assert!(plan.conversation().checkpoint_hints().is_empty());
        journal.compact(&plan).unwrap();
        drop(journal);
        let (_, restored, _) = Journal::resume(&w, "parent-kv", Limits::default()).unwrap();
        assert!(restored.checkpoint_hints().is_empty());
        let mut bad = hint.clone();
        bad.messages = 99999;
        child
            .append_record(
                child.len,
                &[],
                None,
                Some(("engine_checkpoints", crate::kv::hints_value(&[bad]))),
            )
            .unwrap();
        let inspected = child
            .inspect_record(child.record_count() - 1, 0, 4096)
            .unwrap();
        assert!(
            !inspected.contains(&hint.checkpoint.handle),
            "even unusable optional hints must be redacted"
        );
        drop(child);
        let (_, restored, torn) = Journal::resume(&w, "child-kv", Limits::default()).unwrap();
        assert!(!torn);
        assert!(restored.checkpoint_hints().is_empty());
        assert_eq!(
            restored.request(Limits::default().wire).to_json().unwrap(),
            c.request(Limits::default().wire).to_json().unwrap()
        );
    }

    #[test]
    fn forks_preserve_raw_journal_artifacts_and_ancestry_without_sharing_future_messages() {
        use crate::artifacts::Access;
        let f = Fixture::new();
        f.write("file", "before");
        let w = f.workspace();
        let c = Conversation::from_request(json::parse(r#"{"model":"mock","messages":[{"role":"system","content":"stable"},{"role":"user","content":"parent intent"},{"role":"assistant","tool_calls":[{"id":"r","type":"function","function":{"name":"workspace_read","arguments":"{\"path\":\"file\"}"}}]},{"role":"tool","tool_call_id":"r","content":"before"},{"role":"assistant","content":"parent answer"}]}"#).unwrap(), Limits::default()).unwrap();
        let mut parent = Journal::create(&w, "parent", &c).unwrap();
        let id = parent.retain(&"artifact ".repeat(1024)).unwrap();
        let original = std::fs::read(f.0.join(".vade/sessions/parent.journal")).unwrap();
        let (mut child, mut child_c) = parent
            .fork(&w, "child", "explore alternative", Limits::default())
            .unwrap();
        assert_eq!(child_c.messages(), c.messages());
        assert_eq!(child_c.options(), c.options());
        let parent_raw = std::fs::read(f.0.join(".vade/sessions/parent.journal")).unwrap();
        let child_raw = std::fs::read(f.0.join(".vade/sessions/child.journal")).unwrap();
        assert!(parent_raw.starts_with(&original));
        assert!(child_raw.starts_with(&parent_raw));
        assert_eq!(
            parent.read(&id, 0, 1024, 4096).unwrap(),
            child.read(&id, 0, 1024, 4096).unwrap()
        );
        assert!(
            child
                .branch_status(&w)
                .unwrap()
                .contains("parent parent at record")
        );
        assert!(
            child
                .branch_status(&w)
                .unwrap()
                .contains("explore alternative")
        );
        assert!(
            parent
                .fork(&w, "child", "collision", Limits::default())
                .is_err()
        );
        assert_eq!(
            std::fs::read(f.0.join(".vade/sessions/parent.journal")).unwrap(),
            parent_raw
        );
        user(&mut child_c, "child-only intent");
        child.save(&child_c).unwrap();
        let (grandchild, grandchild_c) = child
            .fork(&w, "grandchild", "nested", Limits::default())
            .unwrap();
        assert_eq!(grandchild_c.messages(), child_c.messages());
        assert!(
            grandchild
                .branch_status(&w)
                .unwrap()
                .contains("parent child at record")
        );
        drop(grandchild);
        f.write("file", "changed by child");
        drop(child);
        drop(parent);
        let (parent, restored, _) = Journal::resume(&w, "parent", Limits::default()).unwrap();
        assert_eq!(restored.messages(), c.messages());
        assert!(parent.branch_status(&w).unwrap().contains("1 changed"));
        assert_eq!(f.read("file"), "changed by child");
        let (mut child, restored, _) = Journal::resume(&w, "child", Limits::default()).unwrap();
        assert_eq!(restored.messages(), child_c.messages());
        assert!(child.read(&id, 0, 100, 1000).is_ok());
        assert!(child.fork(&w, "../escape", "", Limits::default()).is_err());
        assert!(child.fork(&w, "child", "", Limits::default()).is_err());
    }
    #[test]
    fn unknown_outcomes_block_forks_and_new_journals_bind_to_workspace_identity() {
        let f = Fixture::new();
        let w = f.workspace();
        let c = Conversation::from_request(json::parse(r#"{"messages":[{"role":"user","content":"request"},{"role":"assistant","tool_calls":[{"id":"x","type":"function","function":{"name":"apply_patch","arguments":"{}"}}]},{"role":"tool","tool_call_id":"x","content":"{\"ok\":false,\"error\":\"unknown_outcome\"}"}]}"#).unwrap(), Limits::default()).unwrap();
        let mut parent = Journal::create(&w, "unknown", &c).unwrap();
        assert!(parent.fork(&w, "child", "", Limits::default()).is_err());
        assert!(!f.0.join(".vade/sessions/child.journal").exists());
        let other = Fixture::new();
        let other_workspace = other.workspace();
        let other_journal = Journal::create(&other_workspace, "copy", &conversation()).unwrap();
        drop(other_journal);
        std::fs::copy(
            f.0.join(".vade/sessions/unknown.journal"),
            other.0.join(".vade/sessions/copy.journal"),
        )
        .unwrap();
        assert!(Journal::resume(&other_workspace, "copy", Limits::default()).is_err());
    }
    #[test]
    fn compaction_records_source_and_preserves_raw_bytes_and_inspection_cursor() {
        let f = Fixture::new();
        let w = f.workspace();
        let c = Conversation::from_request(json::parse(&format!(r#"{{"model":"mock","messages":[{{"role":"system","content":"rules"}},{{"role":"user","content":"constraint"}},{{"role":"assistant","content":"{}"}},{{"role":"user","content":"recent"}},{{"role":"assistant","content":"latest answer"}}]}}"#, "raw-only-long-answer ".repeat(100))).unwrap(), Limits::default()).unwrap();
        let mut journal = Journal::create(&w, "compact", &c).unwrap();
        let path = f.0.join(".vade/sessions/compact.journal");
        let before = std::fs::read(&path).unwrap();
        let plan = crate::context::plan(&c, "Old analysis done.", 1, Limits::default()).unwrap();
        journal.compact(&plan).unwrap();
        assert_eq!(journal.generation(), 1);
        assert_eq!(journal.record_count(), 2);
        assert!(
            journal
                .compact(&plan)
                .unwrap_err()
                .to_string()
                .contains("stale")
        );
        assert_eq!(journal.record_count(), 2);
        assert!(std::fs::read(&path).unwrap().starts_with(&before));
        let raw = journal.inspect_record(0, 0, 4096).unwrap();
        assert!(raw.contains("raw-only-long-answer"));
        assert!(
            !plan
                .conversation()
                .request(Limits::default().wire)
                .to_json()
                .unwrap()
                .contains("raw-only-long-answer")
        );
        let mut active = plan.into_conversation();
        user(&mut active, "after inspection");
        journal.save(&active).unwrap();
        drop(journal);
        let (journal, resumed, torn) = Journal::resume(&w, "compact", Limits::default()).unwrap();
        assert!(!torn);
        assert_eq!(resumed.messages(), active.messages());
        assert_eq!(journal.generation(), 1);
        assert!(journal.compaction_status().contains("source records 0..=0"));
        assert!(journal.compaction_status().contains("Old analysis done."));
        assert!(journal.inspect_record(usize::MAX, 0, 100).is_err());
        assert!(journal.inspect_record(0, usize::MAX, 100).is_err());
    }
    #[test]
    fn artifacts_survive_resume_without_becoming_active_messages() {
        use crate::artifacts::Access;
        let f = Fixture::new();
        let w = f.workspace();
        let c = conversation();
        let mut journal = Journal::create(&w, "artifacts", &c).unwrap();
        let text = "exact \"🦀\" output\n".repeat(1000);
        let id = journal.retain(&text).unwrap();
        assert_eq!(journal.len, c.messages().len());
        drop(journal);
        let (mut journal, resumed, torn) =
            Journal::resume(&w, "artifacts", Limits::default()).unwrap();
        assert!(!torn);
        assert_eq!(resumed.messages(), c.messages());
        let mut offset = 0;
        let mut restored = String::new();
        while offset < text.len() {
            let value = json::parse(&journal.read(&id, offset, 1024, 2048).unwrap()).unwrap();
            restored.push_str(value.get("text").unwrap().as_str().unwrap());
            let Value::Number(next) = value.get("next_offset").unwrap() else {
                panic!()
            };
            offset = next.parse().unwrap();
        }
        assert_eq!(restored, text);
        assert!(journal.read("../other", 0, 1024, 2048).is_err());
        let other = Journal::create(&w, "other", &c).unwrap();
        drop(other);
        let (mut other, _, _) = Journal::resume(&w, "other", Limits::default()).unwrap();
        assert!(other.read(&id, 0, 1024, 2048).is_err());
    }
    #[test]
    fn exact_roundtrip_append_only_and_private_locking() {
        use std::os::unix::fs::PermissionsExt;
        let f = Fixture::new();
        let w = f.workspace();
        let mut c = conversation();
        let mut j = Journal::create(&w, "one", &c).unwrap();
        let path = f.0.join(".vade/sessions/one.journal");
        let first = std::fs::read(&path).unwrap();
        user(&mut c, "exact\n 🦀");
        j.save(&c).unwrap();
        let second = std::fs::read(&path).unwrap();
        assert!(second.starts_with(&first));
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(Journal::resume(&w, "one", Limits::default()).is_err());
        drop(j);
        let (j, restored, torn) = Journal::resume(&w, "one", Limits::default()).unwrap();
        assert!(!torn);
        assert_eq!(restored.messages(), c.messages());
        assert_eq!(restored.options(), c.options());
        drop(j);
        assert!(Journal::create(&w, "one", &c).is_err());
        assert_eq!(list(&w).unwrap(), ["one"]);
    }
    #[test]
    fn every_torn_tail_prefix_recovers_and_can_append() {
        let f = Fixture::new();
        let w = f.workspace();
        let mut c = conversation();
        let mut j = Journal::create(&w, "source", &c).unwrap();
        let path = f.0.join(".vade/sessions/source.journal");
        let initial = std::fs::read(&path).unwrap();
        user(&mut c, "next");
        j.save(&c).unwrap();
        drop(j);
        let complete = std::fs::read(&path).unwrap();
        for cut in initial.len() + 1..complete.len() {
            std::fs::write(&path, &complete[..cut]).unwrap();
            let (mut j, mut recovered, torn) =
                Journal::resume(&w, "source", Limits::default()).unwrap();
            assert!(torn);
            assert_eq!(recovered.messages().len(), 1);
            assert_eq!(std::fs::read(&path).unwrap(), initial);
            user(&mut recovered, "after recovery");
            j.save(&recovered).unwrap();
        }
    }
    #[test]
    fn refuses_corrupt_complete_records_and_redirected_storage() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let f = Fixture::new();
        let w = f.workspace();
        let j = Journal::create(&w, "one", &conversation()).unwrap();
        drop(j);
        let path = f.0.join(".vade/sessions/one.journal");
        let mut bytes = std::fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(Journal::resume(&w, "one", Limits::default()).is_err());
        symlink(&path, f.0.join(".vade/sessions/alias.journal")).unwrap();
        assert!(Journal::resume(&w, "alias", Limits::default()).is_err());
        assert!(Journal::create(&w, "../bad", &conversation()).is_err());
        std::fs::set_permissions(f.0.join(".vade"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        assert!(Journal::create(&w, "new", &conversation()).is_err());
    }
    struct MockModel;
    impl Model for MockModel {
        fn complete(
            &mut self,
            req: ModelRequest<'_>,
            _: &AtomicBool,
            _: &mut dyn FnMut(&str) -> io::Result<()>,
        ) -> io::Result<AssistantTurn> {
            AssistantTurn::from_response(&json::parse(r#"{"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"mock_write","arguments":"{}"}}]}}]}"#).unwrap(), req.wire)
        }
    }
    struct CrashTool;
    impl ToolExecutor for CrashTool {
        fn execute(&mut self, _: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
            panic!("mock crash between intent and result")
        }
    }
    #[test]
    fn crash_during_tool_restores_closed_unknown_outcome_without_replay() {
        let f = Fixture::new();
        let w = f.workspace();
        let mut c = conversation();
        user(&mut c, "write");
        let mut j = Journal::create(&w, "crash", &c).unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            agent::run_observed(
                &mut c,
                &mut MockModel,
                &mut CrashTool,
                Limits::default(),
                &AtomicBool::new(false),
                &mut |_| {},
                &mut j,
            )
        }));
        assert!(panic.is_err());
        drop(j);
        let (_, resumed, torn) = Journal::resume(&w, "crash", Limits::default()).unwrap();
        assert!(!torn);
        assert_eq!(resumed.messages().len(), 4);
        let result = resumed.messages().last().unwrap();
        assert_eq!(result.tool_call_id(), Some("call-1"));
        assert!(
            result
                .value()
                .get("content")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("unknown_outcome")
        );
        // Construction revalidates closed call/result pairing; no executor is
        // part of resume, so the interrupted invocation cannot be replayed.
    }
    #[test]
    fn observer_failure_prevents_any_tool_execution() {
        struct Fail;
        impl RunObserver for Fail {
            fn checkpoint(&mut self, _: &Conversation, _: &[Message]) -> io::Result<()> {
                Err(error("mock disk failure"))
            }
        }
        let mut c = conversation();
        user(&mut c, "write");
        let report = agent::run_observed(
            &mut c,
            &mut MockModel,
            &mut CrashTool,
            Limits::default(),
            &AtomicBool::new(false),
            &mut |_| {},
            &mut Fail,
        );
        assert_eq!(report.reason, agent::StopReason::JournalError);
        assert_eq!(report.tool_calls, 0);
    }
}
