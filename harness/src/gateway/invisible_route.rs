//! Replica routing: fewest gateway in-flight requests, ties rotate. A remembered
//! conversation prefers its previous upstream so engine prefix caches keep
//! hitting. Routing never changes request bytes; identical replicas give identical ids.
//! Cancel and checkpoint release are not routed: they go to every replica (`fan_out`).
use super::{Head, Origin};
use crate::json::{self, View};
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

const STICKY_ENTRIES: usize = 4096;
/// A first turn keeps a remembered upstream only while it is among the least busy:
/// identical concurrent prompts must not pile onto one replica while another idles.
const STICKY_SLACK: usize = 1;
/// A later turn keeps its remembered upstream while that upstream has fewer gateway
/// requests in flight than this, the chats one FP4-Local server admits at the
/// default 32K context (`vade status`: 8 per replica), or while it is among the
/// least busy. Past it the engine queues the turn, and a cold prefill on a free
/// replica can finish sooner.
pub(super) const STICKY_TURN_LIMIT: usize = 8;

/// A chat's routing key, and whether this request continues the chat (any item
/// follows its first user message: a later user turn, a reply or a tool result).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Conversation {
    pub key: u64,
    pub continues: bool,
}

#[derive(Default)]
struct State {
    in_flight: Vec<usize>,
    next: usize,
    clock: u64,
    preferred: HashMap<u64, (usize, u64)>,
}
pub(super) struct Router {
    state: Arc<Mutex<State>>,
    sticky: bool,
}
/// Counts one routed request until the response is delivered or abandoned.
pub(super) struct Lease {
    state: Arc<Mutex<State>>,
    pub index: usize,
}
impl Drop for Lease {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.in_flight[self.index] -= 1;
    }
}
impl Router {
    pub(super) fn new(count: usize, sticky: bool) -> Self {
        Self { state: Arc::new(Mutex::new(State { in_flight: vec![0; count], ..State::default() })), sticky }
    }
    /// Reserve `pinned`, or the least-busy upstream outside `excluded`. Returns None
    /// when every upstream is excluded.
    pub(super) fn acquire(&self, pinned: Option<usize>, conversation: Option<Conversation>, excluded: &[usize]) -> Option<Lease> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let count = state.in_flight.len();
        let open = |index: &usize| !excluded.contains(index);
        let index = match pinned.filter(open) {
            Some(index) => index,
            None => {
                let least = (0..count).filter(open).map(|i| state.in_flight[i]).min()?;
                let bound = |c: Conversation| if c.continues { STICKY_TURN_LIMIT.max(least + STICKY_SLACK) } else { least + STICKY_SLACK };
                let remembered = conversation.filter(|_| self.sticky)
                    .and_then(|c| state.preferred.get(&c.key).map(|entry| (entry.0, bound(c))))
                    .filter(|(i, bound)| *i < count && open(i) && state.in_flight[*i] < *bound).map(|(i, _)| i);
                remembered.unwrap_or_else(|| {
                    let start = state.next;
                    let chosen = (0..count).map(|step| (start + step) % count)
                        .find(|i| open(i) && state.in_flight[*i] == least).unwrap();
                    state.next = (chosen + 1) % count;
                    chosen
                })
            }
        };
        state.in_flight[index] += 1;
        if let Some(key) = conversation.filter(|_| self.sticky && pinned.is_none()).map(|c| c.key) {
            state.clock += 1;
            let clock = state.clock;
            if state.preferred.len() >= STICKY_ENTRIES && !state.preferred.contains_key(&key) {
                let oldest = state.preferred.iter().min_by_key(|(_, entry)| entry.1).map(|(k, _)| *k);
                oldest.map(|k| state.preferred.remove(&k));
            }
            state.preferred.insert(key, (index, clock));
        }
        Some(Lease { state: self.state.clone(), index })
    }
}

/// The chat's key is the `--stable-prefix` cache key, which every turn shares:
/// `prompt_cache_key`, else a session header, else the model, instructions, system,
/// tools and `messages` (or Responses `input`) through the first user message.
/// Requests without a user message in that array carry no conversation.
pub(super) fn conversation_key(body: &[u8], head: &Head) -> Option<Conversation> {
    let value = json::view(std::str::from_utf8(body).ok()?).ok()?;
    let messages = value.get("messages").or_else(|| value.get("input"))?;
    // Tool loops add items with no user role (Chat `tool`, Responses `function_call_output`).
    let mut items = messages.array()?
        .skip_while(|message| message.get("role").and_then(View::as_str).as_deref() != Some("user"));
    items.next()?;
    Some(Conversation { key: super::policy::hash(super::policy::key(Some(value), head).as_bytes()), continues: items.next().is_some() })
}

struct Probe {
    status: String,
    http: Option<u16>,
    model: Option<String>,
    active: u64,
    queued: u64,
    identity: Vec<String>,
    body: String,
}
fn probe(origin: &Origin) -> Probe {
    let mut probe = probe_failed();
    let Ok((code, body)) = fetch_health(origin) else { return probe; };
    probe.http = Some(code);
    let Some(v) = std::str::from_utf8(&body).ok().and_then(|text| json::view(text).ok()).filter(|v| v.fields().is_some()) else {
        probe.status = "invalid".into();
        return probe;
    };
    let text = |v: View<'_>, path: &[&str]| path.iter().try_fold(v, |v, key| v.get(key)).and_then(View::as_str).map(|s| s.into_owned());
    let count = |key| v.get(key).and_then(|n| n.raw().parse::<u64>().ok()).unwrap_or(0);
    probe.status = text(v, &["status"]).unwrap_or_else(|| "invalid".into());
    probe.model = text(v, &["model"]);
    (probe.active, probe.queued) = (count("active"), count("queued"));
    // Build, kernels and checkpoint must agree when both replicas report them.
    probe.identity = [&["vade", "build", "git_sha"][..], &["vade", "ptx_sha256"], &["vade", "checkpoint", "sha256"]]
        .iter().map(|path| text(v, path).unwrap_or_default()).collect();
    probe.body = String::from_utf8(body).unwrap_or_default();
    probe
}
fn fetch_health(origin: &Origin) -> io::Result<(u16, Vec<u8>)> {
    let mut socket = origin.connect()?;
    socket.bound_deadline(Instant::now() + Duration::from_secs(5));
    write!(socket, "GET {} HTTP/1.1\r\nHost: {}\r\nAccept: application/json\r\nConnection: close\r\n", origin.target("/health"), origin.endpoint.authority)?;
    if let Some(key) = crate::auth::ApiKey::configured()? {
        socket.write_all(key.bearer_header().as_bytes())?;
    }
    socket.write_all(b"\r\n")?;
    socket.flush()?;
    let mut reader = BufReader::new(socket);
    let head = loop {
        let head = crate::transport::read_head(&mut reader)?;
        if super::status(&head)? >= 200 { break head; }
    };
    let code = super::status(&head)?;
    let mut body = crate::transport::BodyReader::new(reader, &head, false, 64 * 1024)?;
    let mut bytes = Vec::new();
    body.read_to_end(&mut bytes)?;
    let _ = body.fill_buf();
    Ok((code, bytes))
}

/// One gateway health document: the first ready replica's engine fields with
/// summed load, plus every upstream's state by zero-based position.
pub(super) fn health(origins: &[Origin]) -> (u16, String) {
    let probes: Vec<Probe> = std::thread::scope(|scope| {
        let workers: Vec<_> = origins.iter().map(|origin| scope.spawn(move || probe(origin))).collect();
        workers.into_iter().map(|worker| worker.join().unwrap_or_else(|_| probe_failed())).collect()
    });
    let mut models: Vec<&str> = probes.iter().filter_map(|p| p.model.as_deref()).collect();
    models.sort_unstable();
    models.dedup();
    let identities_differ = (0..3).any(|field| {
        let mut values: Vec<&str> = probes.iter().map(|p| p.identity.get(field).map_or("", String::as_str)).filter(|v| !v.is_empty()).collect();
        values.sort_unstable();
        values.dedup();
        values.len() > 1
    });
    let ready = probes.iter().filter(|p| p.status == "ready" && p.http == Some(200)).count();
    let (code, status, error) = if models.len() > 1 {
        (502, "error", Some("upstream model ids differ"))
    } else if identities_differ {
        (502, "error", Some("upstream engine builds or checkpoints differ"))
    } else if ready == probes.len() {
        (200, "ready", None)
    } else if ready > 0 {
        (503, "degraded", None)
    } else {
        (503, "unavailable", None)
    };
    let first = probes.iter().find(|p| p.status == "ready").and_then(|p| json::view(&p.body).ok());
    let raw = |key| first.and_then(|v| v.get(key)).map_or("null", View::raw);
    let upstreams: Vec<String> = probes.iter().enumerate().map(|(index, p)| format!(
        "{{\"upstream\":{index},\"status\":{},\"http\":{},\"model\":{},\"active\":{},\"queued\":{}}}",
        json::quote(&p.status), p.http.map_or_else(|| "null".into(), |c| c.to_string()),
        p.model.as_deref().map_or_else(|| "null".into(), json::quote), p.active, p.queued)).collect();
    (code, format!(
        "{{\"status\":\"{status}\",\"model\":{},\"active\":{},\"queued\":{},\"vade\":{},\"caps\":{},\"error\":{},\"upstreams\":[{}]}}",
        models.first().map_or_else(|| "null".into(), |m| json::quote(m)),
        probes.iter().map(|p| p.active).sum::<u64>(), probes.iter().map(|p| p.queued).sum::<u64>(),
        raw("vade"), raw("caps"), error.map_or_else(|| "null".into(), json::quote), upstreams.join(",")))
}
fn probe_failed() -> Probe {
    Probe { status: "unreachable".into(), http: None, model: None, active: 0, queued: 0, identity: Vec::new(), body: String::new() }
}
