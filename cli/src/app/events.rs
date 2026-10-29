//! `vade status`: the running server's best-effort events (`GET /v1/events`): a fallback, a
//! retry, a skipped step, a refusal, a pool size decision or a cache miss, with its numbers.
//! The controller (app) and the local launcher print the same section from [`status_lines`].
use super::*;
use crate::evals::json::{self, Value};

impl<R: Runner> App<R> {
    /// The events section. With replicas, each server reports for itself, so each gets its own
    /// labeled section; the gateway serves one of them per request.
    pub(super) fn engine_events(&mut self, session: &Session, health: &str) -> Result<Vec<String>, String> {
        let now = now_ms();
        if !session.replicas { return Ok(status_lines(&self.runner.run(&read(session.endpoint(), "controller events"))?, health, now)); }
        let mut out = Vec::new();
        for index in session.servers() {
            let reply = self.runner.run(&read(format!("http://127.0.0.1:{}", session.tunnel_port(index)), "controller events"))?;
            out.push(format!("{}:", crate::text::terminal_text(&session.ranks[index].machine.name)));
            // A replica's own /health is not read here: a failed read shows no counts.
            out.extend(status_lines(&reply, "", now).into_iter().map(|line| format!("  {line}")));
        }
        Ok(out)
    }
}

/// The `GET /v1/events` read; `endpoint` has no trailing slash.
pub(crate) fn read(endpoint: String, label: &str) -> Exec {
    super::health::get(label, format!("{endpoint}/v1/events"))
}
pub(crate) fn now_ms() -> i64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64) }

/// What `vade status` prints from the `GET /v1/events` reply. A failed or bad read is not
/// "unsupported": an engine that has the route also puts `"events"` counts in its ready
/// `/health` body, so those counts are kept and the failed read is named. `health` is that
/// body, or empty when it was not read.
pub(crate) fn status_lines(reply: &Output, health: &str, now_ms: i64) -> Vec<String> {
    let parsed = if reply.success { section(&reply.stdout, now_ms) } else { Err("the request failed".into()) };
    let error = match parsed { Ok(lines) => return lines, Err(error) => crate::text::terminal_text(&error).into_owned() };
    let health = json::parse(health).ok();
    match health.as_ref().and_then(|value| value.get("events")).map(counts) {
        Some(Some(counts)) => vec![format!("Event counts: {counts} (from /health)"), format!("Last events: not read ({error}; run vade status again)")],
        None if health.is_some() && !reply.success => vec!["Events: not reported by this engine".into()],
        _ => vec![format!("Events: not read ({error}; run vade status again)")],
    }
}

/// Lines `vade status` prints from the server's last events.
pub(crate) const SHOWN: usize = 20;

/// A `{name:count}` object as `name count, ...` in the server's (name) order; `None` if it is not one.
pub(crate) fn counts(counts: &Value) -> Option<String> {
    let Value::Object(counts) = counts else { return None; };
    if counts.is_empty() { return Some("none".into()); }
    let mut out = Vec::with_capacity(counts.len());
    for (name, count) in counts { out.push(format!("{} {}", crate::text::terminal_text(name), count.as_i64()?)); }
    Some(out.join(", "))
}

/// The last `shown` events, oldest first, as `HH:MM:SSZ (age) line`. Times are UTC from the
/// server's clock; the age uses `now_ms` from this machine's clock.
pub(crate) fn recent(value: &Value, shown: usize, now_ms: i64) -> Option<Vec<String>> {
    let events = value.get("recent")?.as_array()?;
    let mut out = Vec::with_capacity(shown.min(events.len()));
    for event in &events[events.len().saturating_sub(shown)..] {
        let at = event.get("unix_ms")?.as_i64()?;
        let line = event.get("line")?.as_str()?;
        let day = at.rem_euclid(86_400_000) / 1000;
        out.push(format!("{:02}:{:02}:{:02}Z ({}) {}", day / 3600, day / 60 % 60, day % 60, age(now_ms - at),
            crate::text::terminal_text(&line.replace('\n', " | "))));
    }
    Some(out)
}
fn age(ms: i64) -> String {
    let s = ms.max(0) / 1000;
    if s < 120 { format!("{s} s ago") } else if s < 7200 { format!("{} min ago", s / 60) } else { format!("{} h ago", s / 3600) }
}

/// The section `vade status` prints, from the `GET /v1/events` reply body.
pub(crate) fn section(body: &str, now_ms: i64) -> Result<Vec<String>, String> {
    let value = json::parse(body)?;
    let counts = value.get("counts").and_then(counts).ok_or("the events reply has no counts")?;
    let recent = recent(&value, SHOWN, now_ms).ok_or("the events reply has no recent list")?;
    let mut out = vec![format!("Event counts: {counts}")];
    if recent.is_empty() { out.push("Last events: none".into()); }
    else { out.push(format!("Last {} events (oldest first):", recent.len())); out.extend(recent.into_iter().map(|line| format!("  {line}"))); }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn body(events: usize) -> String {
        let recent: Vec<_> = (0..events).map(|i| format!("{{\"unix_ms\":{},\"name\":\"prefix_cache_miss\",\"line\":\"prefix_cache_miss prompt_tokens={i} reason=empty\"}}", 1_793_000_000_000i64 + i as i64 * 1000)).collect();
        format!("{{\"counts\":{{\"prefix_cache_miss\":{events},\"scheduler_pool\":1}},\"recent\":[{}]}}", recent.join(","))
    }
    #[test]
    fn prints_counts_and_the_last_twenty_events_oldest_first() {
        let now = 1_793_000_000_000 + 64 * 1000 + 500;
        let lines = section(&body(64), now).unwrap();
        assert_eq!(lines[0], "Event counts: prefix_cache_miss 64, scheduler_pool 1");
        assert_eq!(lines[1], "Last 20 events (oldest first):");
        assert_eq!(lines.len(), 2 + SHOWN);
        // 1_793_000_000_000 ms is 07:33:20 UTC (date -u -r 1793000000); event 44 is 44 s later and 20.5 s before now.
        assert_eq!(lines[2], "  07:34:04Z (20 s ago) prefix_cache_miss prompt_tokens=44 reason=empty");
        assert_eq!(lines[21], "  07:34:23Z (1 s ago) prefix_cache_miss prompt_tokens=63 reason=empty");
    }
    #[test]
    fn short_and_empty_lists_and_bad_replies() {
        let lines = section(&body(2), 1_793_000_000_000 + 600_000).unwrap();
        assert_eq!(lines[1], "Last 2 events (oldest first):");
        assert!(lines[2].contains("(10 min ago) prefix_cache_miss prompt_tokens=0"), "{lines:?}");
        assert_eq!(section("{\"counts\":{},\"recent\":[]}", 0).unwrap(), ["Event counts: none", "Last events: none"]);
        for bad in ["", "{}", "{\"counts\":{}}", "{\"counts\":{\"a\":\"x\"},\"recent\":[]}", "{\"counts\":{},\"recent\":[{\"line\":\"a\"}]}"] {
            assert!(section(bad, 0).is_err(), "{bad}");
        }
        assert_eq!(age(-5), "0 s ago");
        assert_eq!(age(3 * 3600 * 1000), "3 h ago");
    }
    #[test]
    fn a_failed_read_is_not_an_older_engine() {
        let failed = Output { success: false, stdout: String::new() };
        let ok = |body: &str| Output { success: true, stdout: body.into() };
        let health = r#"{"status":"ready","events":{"admission_refused":3}}"#;
        // The engine has the route (its /health carries counts): keep the counts, name the failure.
        assert_eq!(status_lines(&failed, health, 0), ["Event counts: admission_refused 3 (from /health)", "Last events: not read (the request failed; run vade status again)"]);
        assert_eq!(status_lines(&ok("{}"), health, 0)[1], "Last events: not read (the events reply has no counts; run vade status again)");
        // An older engine has no counts in /health and refuses the route.
        assert_eq!(status_lines(&failed, r#"{"status":"ready"}"#, 0), ["Events: not reported by this engine"]);
        // No /health body to tell the two apart: say only that the read failed.
        assert_eq!(status_lines(&failed, "", 0), ["Events: not read (the request failed; run vade status again)"]);
        assert_eq!(status_lines(&ok(&body(1)), "", 0)[0], "Event counts: prefix_cache_miss 1, scheduler_pool 1");
    }
    #[test]
    fn server_text_is_made_safe_for_the_terminal() {
        let body = "{\"counts\":{\"x\":1},\"recent\":[{\"unix_ms\":0,\"name\":\"x\",\"line\":\"x error=a\\u001b[31m\\nb\"}]}";
        let lines = section(body, 0).unwrap();
        assert!(!lines[2].contains('\u{1b}') && lines[2].ends_with(" | b"), "{lines:?}");
    }
}
