//! Best-effort events: a fallback, a retry, a skipped step, a refusal, a sizing decision or a
//! cache miss. Each one writes one log line that starts with its stable name, adds one to that
//! name's count and keeps the line among the last [`RECENT`]. This is host bookkeeping only:
//! nothing reads it back into a decision, so serving output cannot depend on it.
use std::{collections::{BTreeMap, VecDeque}, fmt, io::Write, sync::Mutex, time::{SystemTime, UNIX_EPOCH}};

/// Lines kept for `GET /v1/events`.
pub const RECENT: usize = 64;
/// A kept line is cut at this many bytes (on a character boundary); the log keeps all of it.
pub const LINE_BYTES: usize = 480;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event { pub unix_ms: u64, pub name: &'static str, pub line: String }

#[derive(Debug, Default)]
pub struct Events { counts: BTreeMap<&'static str, u64>, recent: VecDeque<Event> }
impl Events {
    pub const fn new() -> Self { Self { counts: BTreeMap::new(), recent: VecDeque::new() } }
    pub fn record(&mut self, name: &'static str, line: &str, unix_ms: u64) {
        let count = self.counts.entry(name).or_insert(0);
        *count = count.saturating_add(1);
        if self.recent.len() == RECENT { self.recent.pop_front(); }
        let mut end = line.len().min(LINE_BYTES);
        while !line.is_char_boundary(end) { end -= 1; }
        self.recent.push_back(Event { unix_ms, name, line: line[..end].to_owned() });
    }
    pub fn count(&self, name: &str) -> u64 { self.counts.get(name).copied().unwrap_or(0) }
    pub fn recent(&self) -> impl Iterator<Item = &Event> { self.recent.iter() }
    /// `{"name":count,...}` in name order.
    pub fn counts_json(&self) -> String {
        self.counts_json_filtered(|_| true)
    }
    /// Counts for names the caller can expose. Stored counts and logs do not change.
    pub fn counts_json_filtered(&self, include: impl Fn(&str) -> bool) -> String {
        let mut out = String::from("{");
        for (i, (name, count)) in self.counts.iter().filter(|(name, _)| include(name)).enumerate() {
            if i != 0 { out.push(','); }
            quote_into(&mut out, name);
            out.push_str(&format!(":{count}"));
        }
        out.push('}');
        out
    }
    /// `{"counts":{...},"recent":[{"unix_ms":..,"name":..,"line":..},...]}`, oldest line first.
    pub fn json(&self) -> String {
        self.json_filtered(|_| true)
    }
    /// Apply the same name filter to counts and raw lines, including their timestamps.
    pub fn json_filtered(&self, include: impl Fn(&str) -> bool) -> String {
        let mut out = format!("{{\"counts\":{},\"recent\":[", self.counts_json_filtered(&include));
        for (i, event) in self.recent.iter().filter(|event| include(event.name)).enumerate() {
            if i != 0 { out.push(','); }
            out.push_str(&format!("{{\"unix_ms\":{},\"name\":", event.unix_ms));
            quote_into(&mut out, event.name);
            out.push_str(",\"line\":");
            quote_into(&mut out, &event.line);
            out.push('}');
        }
        out.push_str("]}");
        out
    }
}

static EVENTS: Mutex<Events> = Mutex::new(Events::new());
fn with<T>(f: impl FnOnce(&mut Events) -> T) -> T { f(&mut EVENTS.lock().unwrap_or_else(|e| e.into_inner())) }
fn now_ms() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64) }

/// Log `name k=v ...` on stderr as one line (a multi-line error joins with " | ") and count it under `name`.
pub fn note(name: &'static str, fields: fmt::Arguments) { note_line(name, &one_line(format!("{name} {fields}"))); }
fn one_line(line: String) -> String { if line.contains('\n') { line.trim_end().replace('\n', " | ") } else { line } }
/// For a log line kept as it was before this module: counted under `name`, printed unchanged.
pub fn note_line(name: &'static str, line: &str) { note_to(&mut std::io::stderr(), name, line); }
/// Count first, then write. A failed log write (a closed pipe, a full disk) is ignored: an event
/// can come from a destructor, and a panic there during unwinding aborts the process.
fn note_to(log: &mut impl Write, name: &'static str, line: &str) {
    with(|events| events.record(name, line, now_ms()));
    let _ = writeln!(log, "{line}");
}
/// This process's counts so far, for a receipt.
pub fn counts_json() -> String { with(|events| events.counts_json()) }
/// Counts for names the caller can expose, without changing the registry or log.
pub fn counts_json_filtered(include: impl Fn(&str) -> bool) -> String { with(|events| events.counts_json_filtered(include)) }
/// This process's counts and last lines, for `GET /v1/events`.
pub fn json() -> String { with(|events| events.json()) }
/// Counts and last lines for names the caller can expose.
pub fn json_filtered(include: impl Fn(&str) -> bool) -> String { with(|events| events.json_filtered(include)) }
pub fn count(name: &str) -> u64 { with(|events| events.count(name)) }

fn quote_into(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn filtering_hides_names_counts_and_lines_without_removing_records() {
        let mut events = Events::new();
        for name in ["prefix_cache_miss", "prefix_cache_status", "prefix_cache_make_room", "prefix_cache_future"] {
            events.record(name, &format!("{name} tokens=123 retained_bytes=456"), 789);
        }
        events.record("scheduler_pool", "scheduler_pool ready", 12);
        let visible = |name: &str| !name.starts_with("prefix_cache_");
        assert_eq!(events.counts_json_filtered(visible), r#"{"scheduler_pool":1}"#);
        assert_eq!(events.json_filtered(visible), r#"{"counts":{"scheduler_pool":1},"recent":[{"unix_ms":12,"name":"scheduler_pool","line":"scheduler_pool ready"}]}"#);
        assert_eq!(events.json_filtered(|_| false), r#"{"counts":{},"recent":[]}"#);
        assert_eq!(events.count("prefix_cache_miss"), 1);
        assert_eq!(events.recent().count(), 5);
        assert!(events.json().contains("tokens=123 retained_bytes=456"));
    }
    #[test]
    fn counts_by_name_and_keeps_the_last_lines() {
        let mut events = Events::new();
        assert_eq!(events.counts_json(), "{}");
        assert_eq!(events.json(), "{\"counts\":{},\"recent\":[]}");
        events.record("cache_miss", "cache_miss reason=scope_miss tokens=12", 5);
        events.record("admission_refused", "admission_refused bytes=7", 6);
        events.record("cache_miss", "cache_miss reason=no_restorable_prefix tokens=3", 7);
        assert_eq!((events.count("cache_miss"), events.count("admission_refused"), events.count("other")), (2, 1, 0));
        assert_eq!(events.counts_json(), "{\"admission_refused\":1,\"cache_miss\":2}");
        assert!(events.json().ends_with("{\"unix_ms\":7,\"name\":\"cache_miss\",\"line\":\"cache_miss reason=no_restorable_prefix tokens=3\"}]}"));
        for i in 0..RECENT as u64 + 5 { events.record("retry", &format!("retry n={i}"), i); }
        assert_eq!(events.recent().count(), RECENT);
        assert_eq!(events.recent().next().unwrap().line, "retry n=5");
        assert_eq!(events.recent().last().unwrap().line, format!("retry n={}", RECENT + 4));
        assert_eq!(events.count("retry"), RECENT as u64 + 5);
        assert_eq!(events.count("cache_miss"), 2, "counts outlive the kept lines");
    }
    #[test]
    fn lines_are_cut_on_a_character_and_quoted_as_json() {
        let mut events = Events::new();
        let long = format!("{}é tail", "x".repeat(LINE_BYTES - 1));
        events.record("long", &long, 1);
        let kept = &events.recent().next().unwrap().line;
        assert_eq!(kept.len(), LINE_BYTES - 1, "the two-byte character at the cut is dropped whole");
        events.record("quoted", "quoted error=\"a\\b\"\nnext\u{1}", 2);
        let json = events.json();
        assert!(json.contains("\"line\":\"quoted error=\\\"a\\\\b\\\"\\nnext\\u0001\""), "{json}");
        let parsed = crate::json::parse(json.as_bytes(), Default::default()).unwrap();
        let recent = parsed.object().unwrap()["recent"].array().unwrap();
        assert_eq!(recent[1].object().unwrap()["line"].string().unwrap(), "quoted error=\"a\\b\"\nnext\u{1}");
    }
    #[test]
    fn the_process_registry_counts_each_note() {
        let before = count("events_self_test");
        note("events_self_test", format_args!("n={}", 1));
        note_line("events_self_test", "legacy text kept unchanged");
        assert_eq!(one_line("refused Requested size does not fit.\nGap: 2 bytes.\n".into()), "refused Requested size does not fit. | Gap: 2 bytes.");
        assert_eq!(count("events_self_test"), before + 2);
        assert!(counts_json().contains("\"events_self_test\":"));
        assert!(json().contains("legacy text kept unchanged"));
    }
    #[test]
    fn a_failed_log_write_still_counts_and_never_panics() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> { Err(std::io::ErrorKind::BrokenPipe.into()) }
            fn flush(&mut self) -> std::io::Result<()> { Err(std::io::ErrorKind::BrokenPipe.into()) }
        }
        let before = count("events_broken_log_test");
        note_to(&mut Broken, "events_broken_log_test", "events_broken_log_test n=1");
        assert_eq!(count("events_broken_log_test"), before + 1);
        let mut kept = Vec::new();
        note_to(&mut kept, "events_broken_log_test", "events_broken_log_test n=2");
        assert_eq!((kept.as_slice(), count("events_broken_log_test")), (&b"events_broken_log_test n=2\n"[..], before + 2));
    }
}
