//! Native checkpoint lifecycle. A shared, bounded owner keeps parent handles
//! alive across branch clients; children never release their parent's reference.
use super::*;
use crate::kv::{self, Capabilities, Checkpoint, Hint};
use std::collections::{BTreeMap, VecDeque};

const MAX_BRANCHES: usize = 128;
const RELEASE_BUDGET: Duration = Duration::from_secs(2);

/// How many checkpoint releases were attempted and how many the engine
/// confirmed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReleaseSummary {
    pub attempted: usize,
    pub released: usize,
}
#[derive(Default)]
struct State {
    handles: VecDeque<Checkpoint>,
    branches: BTreeMap<String, Vec<String>>,
}
pub(super) struct Session {
    endpoint: Endpoint,
    pool: ConnectionPool,
    state: Mutex<State>,
}
impl Session {
    pub(super) fn new(endpoint: Endpoint) -> Self {
        Self {
            endpoint,
            pool: ConnectionPool::default(),
            state: Mutex::new(State::default()),
        }
    }
    pub(super) fn can_track(&self, branch: &str) -> bool {
        self.state
            .lock()
            .is_ok_and(|s| s.branches.contains_key(branch) || s.branches.len() < MAX_BRANCHES)
    }
    /// Register only hints validated against the current negotiated scope. The
    /// Engine enforces actual shared-byte quotas; summing handle sizes is wrong.
    pub(super) fn remember(&self, branch: &str, hints: &[Hint], caps: &Capabilities) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if !state.branches.contains_key(branch) && state.branches.len() >= MAX_BRANCHES {
            return; // bounded bookkeeping; untracked hints can expire normally
        }
        let at = kv::now_ms();
        let keys: Vec<_> = hints
            .iter()
            .filter(|h| h.checkpoint.eligible(caps, at))
            .map(|h| h.checkpoint.handle.clone())
            .collect();
        state.branches.insert(branch.to_owned(), keys);
        for hint in hints.iter().filter(|h| h.checkpoint.eligible(caps, at)) {
            state.handles.retain(|c| c.handle != hint.checkpoint.handle);
            state.handles.push_back(hint.checkpoint.clone());
        }
        let mut release = Vec::new();
        let mut kept = VecDeque::new();
        while let Some(c) = state.handles.pop_front() {
            if c.expires_at_ms <= at {
                continue;
            }
            if c.cache_domain != caps.cache_domain
                || !state.branches.values().any(|v| v.contains(&c.handle))
            {
                release.push(c);
            } else {
                kept.push_back(c);
            }
        }
        while kept.len() > caps.limit() {
            // Pressure is normal: journal hints remain expiring best-effort
            // references; a later request still sends its entire transcript.
            release.push(kept.pop_front().unwrap());
        }
        state.handles = kept;
        drop(state);
        let _ = self.release(release);
    }
    pub(super) fn forget(
        &self,
        branch: &str,
        hints: &[Hint],
        caps: Option<&Capabilities>,
    ) -> ReleaseSummary {
        let Ok(mut state) = self.state.lock() else {
            return ReleaseSummary::default();
        };
        state.branches.remove(branch);
        let mut release = Vec::new();
        // Include a just-deleted journal's restored hints, but only for the
        // negotiated authorization domain and only when no known branch uses it.
        for hint in hints.iter().take(kv::MAX_HINTS) {
            if caps.is_some_and(|c| hint.checkpoint.eligible(c, kv::now_ms()))
                && !state
                    .handles
                    .iter()
                    .any(|c| c.handle == hint.checkpoint.handle)
            {
                state.handles.push_back(hint.checkpoint.clone());
            }
        }
        let mut kept = VecDeque::new();
        while let Some(c) = state.handles.pop_front() {
            if state.branches.values().any(|v| v.contains(&c.handle)) {
                kept.push_back(c);
            } else {
                release.push(c);
            }
        }
        state.handles = kept;
        drop(state);
        self.release(release)
    }
    pub(super) fn release_all(&self) -> ReleaseSummary {
        let Ok(mut state) = self.state.lock() else {
            return ReleaseSummary::default();
        };
        state.branches.clear();
        let release = state.handles.drain(..).collect();
        drop(state);
        self.release(release)
    }
    fn release(&self, checkpoints: Vec<Checkpoint>) -> ReleaseSummary {
        let mut summary = ReleaseSummary::default();
        let Some(endpoint) = crate::contract::sibling(&self.endpoint, "kv/release") else {
            return summary;
        };
        let deadline = Instant::now() + RELEASE_BUDGET;
        for checkpoint in checkpoints
            .into_iter()
            .take(kv::MAX_HANDLES + kv::MAX_HINTS)
        {
            if checkpoint.expires_at_ms <= kv::now_ms() || Instant::now() >= deadline {
                continue;
            }
            summary.attempted += 1;
            let body = format!("{{\"kv_handle\":{}}}", json::quote(&checkpoint.handle));
            let result = (|| -> io::Result<()> {
                let mut response = request_method_pooled_deadline(
                    "POST",
                    &endpoint,
                    body.as_bytes(),
                    None,
                    None,
                    &self.pool,
                    Some(deadline),
                )?;
                if response.status != 200 {
                    return Err(invalid("KV release unavailable"));
                }
                let mut bytes = Vec::new();
                response.body.by_ref().take(4097).read_to_end(&mut bytes)?;
                if bytes.len() > 4096 {
                    return Err(invalid("KV release exceeds limit"));
                }
                let v = json::view(
                    std::str::from_utf8(&bytes).map_err(|_| invalid("KV release encoding"))?,
                )?;
                if v.get("released").is_none_or(|v| v.raw() != "true") {
                    return Err(invalid("KV release unconfirmed"));
                }
                response.recycle();
                Ok(())
            })();
            summary.released += usize::from(result.is_ok());
        }
        summary
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.release_all();
    }
}

pub(super) fn managed(request: ModelRequest<'_>, caps: Option<&Capabilities>) -> bool {
    caps.is_some()
        && request.options.get("cache") != Some(&json::Value::Bool(false))
        && request.options.get("kv_handle").is_none()
        && request.options.get("kv_checkpoint").is_none()
}
/// Request-local hints come AFTER the immutable options/messages. The canonical
/// append-only builder never sees a changing handle, and admission sees exactly
/// the same final bytes as generation.
pub(super) fn decorate(body: &[u8], handle: Option<&Checkpoint>) -> io::Result<Vec<u8>> {
    let mut suffix = String::from(",\"kv_checkpoint\":true");
    if let Some(checkpoint) = handle {
        suffix.push_str(",\"kv_handle\":");
        json::write_string(&checkpoint.handle, &mut suffix);
    }
    suffix.push('}');
    if body.last() != Some(&b'}') || body.len() + suffix.len() - 1 > json::MAX_INPUT_BYTES {
        return Err(invalid("checkpoint request exceeds byte limit"));
    }
    let mut result = Vec::with_capacity(body.len() + suffix.len() - 1);
    result.extend_from_slice(&body[..body.len() - 1]);
    result.extend_from_slice(suffix.as_bytes());
    Ok(result)
}
pub(super) fn eligible(hints: &[Hint], caps: &Capabilities, messages: usize) -> Vec<Hint> {
    hints
        .iter()
        .filter(|h| h.messages <= messages && h.checkpoint.eligible(caps, kv::now_ms()))
        .take(kv::MAX_HINTS)
        .cloned()
        .collect()
}
pub(super) fn update(
    hints: &mut Vec<Hint>,
    selected: Option<&Checkpoint>,
    report: Option<&crate::kv::Report>,
    caps: &Capabilities,
    messages: usize,
) {
    let Some(report) = report.filter(|r| r.cache_domain == caps.cache_domain) else {
        return;
    };
    if report.reuse.rejected() {
        hints.retain(|h| selected.is_none_or(|s| h.checkpoint.handle != s.handle));
    }
    if let Some(checkpoint) = &report.checkpoint {
        if checkpoint.eligible(caps, kv::now_ms()) {
            let mut checkpoint = checkpoint.clone();
            // A response cannot grant a longer local lease than advertised.
            checkpoint.expires_at_ms = checkpoint
                .expires_at_ms
                .min(kv::now_ms().saturating_add(caps.ttl_ms));
            hints.retain(|h| h.checkpoint.handle != checkpoint.handle && h.messages != messages);
            hints.push(Hint {
                checkpoint,
                messages,
            });
        }
    }
    hints.sort_by_key(|h| h.messages);
    let limit = kv::MAX_HINTS.min(caps.limit());
    while hints.len() > limit {
        // Preserve earliest usable input plus latest input when quota permits.
        hints.remove(if limit > 1 { 1 } else { 0 });
    }
}
