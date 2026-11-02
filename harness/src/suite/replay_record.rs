//! Read the proxy's recorded replies before a replay starts.
use super::{serve::Reply, task::Task, u64_of};
use crate::agent_wire::{AssistantTurn, ChatAccumulator, WireLimits};
use crate::json::{self, Value};

struct Record { line: usize, step: usize, call: usize, held: bool, reply: Reply }

pub(super) struct Recording { pub replies: Vec<Reply>, pub calls: Vec<(usize, usize, u16)> }

/// The proxy can save a held call after a later call. Sort the records by
/// step and call, then check the sequence before returning the replies.
pub(super) fn parse(text: &str, task: &Task) -> Result<Recording, String> {
    let mut records = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        records.push(record(line).map_err(|e| format!("line {}: {e}", i + 1)).map(|mut r| { r.line = i + 1; r })?);
    }
    if records.is_empty() { return Err("the transcript has no recorded calls".into()); }
    records.sort_by_key(|r| (r.step, r.call));
    let steps = if task.mode == "direct" { 1 } else { task.steps.len() };
    let mut at = 0;
    for step in 1..=steps {
        let held_at = task.steps.get(step - 1).and_then(|s| s.cancel_at_call);
        let start = at;
        let mut call = 1;
        while let Some(r) = records.get(at).filter(|r| r.step == step) {
            let error = |e: String| format!("line {}, step {step}, call {}: {e}", r.line, r.call);
            if r.call != call { return Err(error(format!("expected call {call}; a call is missing or repeated"))); }
            if r.held != (held_at == Some(call)) { return Err(error("held does not match task.txt cancel_at_call".into())); }
            if r.held && records.get(at + 1).is_some_and(|next| next.step == step) {
                return Err(error("a held call must be the last call of its step".into()));
            }
            call += 1;
            at += 1;
        }
        if at == start { return Err(format!("step {step}: the transcript has no recorded calls")); }
        let last = &records[at - 1];
        if held_at.is_some() && !last.held { return Err(format!("step {step}: the transcript is missing the held call")); }
    }
    if let Some(r) = records.get(at) { return Err(format!("line {}: step {} is not in task.txt", r.line, r.step)); }
    let calls = records.iter().map(|r| (r.step, r.call, r.reply.status)).collect();
    let replies = records.into_iter().filter(|r| !r.held).map(|r| r.reply).collect();
    Ok(Recording { replies, calls })
}

fn record(text: &str) -> Result<Record, String> {
    let v = json::parse(text).map_err(|e| format!("invalid record JSON: {e}"))?;
    let number = |key: &str| v.get(key).and_then(u64_of).ok_or_else(|| format!("{key} must be an unsigned integer"));
    let string = |key: &str| v.get(key).and_then(Value::as_str).ok_or_else(|| format!("{key} must be a string"));
    let boolean = |key: &str| match v.get(key) { Some(Value::Bool(b)) => Ok(*b), _ => Err(format!("{key} must be true or false")) };
    let positive = |key: &str| -> Result<usize, String> {
        usize::try_from(number(key)?).ok().filter(|n| *n > 0).ok_or_else(|| format!("{key} must be a positive integer"))
    };
    let (step, call) = (positive("step")?, positive("call")?);
    let (held, client_closed) = (boolean("held")?, boolean("client_closed")?);
    let status = u16::try_from(number("status")?).map_err(|_| "status is outside the HTTP range")?;
    let content_type = string("content_type")?;
    if content_type.contains(['\r', '\n']) { return Err("content_type contains a line break".into()); }
    let request = json::parse(string("request")?).map_err(|e| format!("invalid request JSON: {e}"))?;
    if !matches!(request, Value::Object(_)) { return Err("request JSON must be an object".into()); }
    let body = string("response")?;
    if held {
        if status != 0 || !body.is_empty() || !content_type.is_empty() { return Err("a held call must have status 0 and no response".into()); }
        if !client_closed { return Err("the held call was not closed by the client".into()); }
    } else {
        if !(200..=599).contains(&status) { return Err("status must be a final HTTP status from 200 to 599".into()); }
        if (200..300).contains(&status) { response(content_type, body)?; }
    }
    Ok(Record { line: 0, step, call, held, reply: Reply { status, content_type: content_type.into(), body: body.into() } })
}

/// Use the same wire checks as the harness. A valid SSE body has complete
/// events, a finish reason and [DONE]. Data after [DONE] is an error.
fn response(content_type: &str, body: &str) -> Result<(), String> {
    let kind = content_type.split(';').next().unwrap_or("").trim();
    if kind.eq_ignore_ascii_case("application/json") {
        let v = json::parse(body).map_err(|e| format!("invalid response JSON: {e}"))?;
        AssistantTurn::from_response(&v, WireLimits::default()).map_err(|e| format!("invalid response: {e}"))?;
    } else if kind.eq_ignore_ascii_case("text/event-stream") {
        let mut decoder = crate::transport::SseDecoder::default();
        let mut assembly = ChatAccumulator::new(WireLimits::default());
        decoder.feed(body.as_bytes(), &mut |data| { assembly.push_sse_data(data, &mut |_| Ok(()))?; Ok(false) })
            .map_err(|e| format!("invalid response stream: {e}"))?;
        if !decoder.at_event_boundary() { return Err("incomplete response stream event".into()); }
        assembly.into_turn().map_err(|e| format!("incomplete response stream: {e}"))?;
    } else {
        return Err("a successful response needs application/json or text/event-stream".into());
    }
    // A run can stop after a tool call when it reaches a harness limit.
    // The replay checks the actual call sequence to detect missing replies.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{n, obj, s, task, write};

    fn task() -> Task { task::parse("id: record\ncategory: document-reading\nstep: new\nprompt: p\ncheck: last-exit-zero\n").unwrap() }

    fn entry(step: usize, call: usize, body: &str, content_type: &str, held: bool) -> String {
        write(&obj(vec![("step", n(step)), ("call", n(call)), ("status", n(if held { 0 } else { 200 })),
            ("held", Value::Bool(held)), ("client_closed", Value::Bool(held)), ("request", s("{\"messages\":[]}")),
            ("response", s(body)), ("content_type", s(content_type))]))
    }

    fn answer(step: usize, call: usize) -> String { entry(step, call, &task::answer_reply(call, "Done."), "application/json", false) }

    #[test]
    fn reads_json_and_streamed_replies() {
        let raw = task::parse_script(">>> call workspace_read\n{\"path\":\"note.txt\"}\n").unwrap().remove(0).body;
        let stream = super::super::serve::to_sse(&task::answer_reply(2, "Done."), 16).concat();
        let text = format!("{}\n{}\n", entry(1, 1, &raw, "application/json", false), entry(1, 2, &stream, "text/event-stream; charset=utf-8", false));
        let record = parse(&text, &task()).unwrap();
        assert_eq!(record.calls, [(1, 1, 200), (1, 2, 200)]);
        assert_eq!(record.replies.len(), 2);
        assert_eq!(record.replies[0].body, raw);
        assert_eq!(record.replies[1].body, stream);
    }

    #[test]
    fn sorts_late_held_calls_and_skips_their_replies() {
        let task = task::parse("id: record\ncategory: session-resume\nstep: cancel\nprompt: p\ncancel_at_call: 2\nstep: resume\nprompt: p\ncheck: held-closed\n").unwrap();
        let text = format!("{}\n{}\n{}\n", answer(1, 1), answer(2, 1), entry(1, 2, "", "", true));
        let record = parse(&text, &task).unwrap();
        assert_eq!(record.replies.len(), 2);
        assert_eq!(record.calls, [(1, 1, 200), (1, 2, 0), (2, 1, 200)]);
        assert!(parse(&format!("{}\n{}", answer(1, 1), answer(2, 1)), &task).err().unwrap().contains("missing the held call"));
        let bad = format!("{}\n{}\n{}", answer(1, 1), answer(1, 2), answer(2, 1));
        assert!(parse(&bad, &task).err().unwrap().contains("held does not match"));
    }

    #[test]
    fn rejects_missing_and_repeated_calls() {
        for (text, error) in [(String::new(), "no recorded calls"), (answer(1, 2), "expected call 1"),
            (format!("{}\n{}", answer(1, 1), answer(1, 1)), "expected call 2"), (answer(2, 1), "step 1")] {
            assert!(parse(&text, &task()).err().unwrap().contains(error), "{error}");
        }
        let raw = task::parse_script(">>> call workspace_read\n{\"path\":\"note.txt\"}\n").unwrap().remove(0).body;
        assert!(parse(&entry(1, 1, &raw, "application/json", false), &task()).is_ok());
    }

    #[test]
    fn rejects_bad_records_with_line_numbers() {
        for text in ["{", "{}", "[]", "{\"step\":-1}", "{\"step\":18446744073709551616}"] {
            assert!(parse(text, &task()).err().unwrap().starts_with("line 1:"));
        }
        let wrong = answer(1, 1).replace("\"held\":false", "\"held\":0");
        assert!(parse(&wrong, &task()).err().unwrap().contains("held must be true or false"));
        let wrong = answer(1, 1).replace("{\\\"messages\\\":[]}", "[]");
        assert!(parse(&wrong, &task()).err().unwrap().contains("request JSON must be an object"));
        let wrong = entry(1, 1, "{}", "application/json\r\nExtra: x", false);
        assert!(parse(&wrong, &task()).err().unwrap().contains("content_type contains a line break"));
    }

    #[test]
    fn rejects_broken_successful_replies() {
        let complete = "data: {\"choices\":[{\"delta\":{\"content\":\"Done.\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let incomplete = complete.replace("data: [DONE]\n\n", "");
        let no_finish = complete.replace("\"stop\"", "null");
        let extra = format!("{complete}data: {{}}\n\n");
        for stream in ["data: {\n\n", incomplete.as_str(), no_finish.as_str(), complete.trim_end(), extra.as_str()] {
            assert!(parse(&entry(1, 1, stream, "text/event-stream", false), &task()).is_err(), "accepted incomplete stream");
        }
        for body in ["{", "{}", "[]", "{\"choices\":[]}"] {
            assert!(parse(&entry(1, 1, body, "application/json", false), &task()).is_err());
        }
        let valid = complete.replace("\n", "\r\n");
        assert!(parse(&entry(1, 1, &valid, "text/event-stream", false), &task()).is_ok());
    }

    #[test]
    fn preserves_http_error_status() {
        let text = entry(1, 1, "{\"error\":\"unsupported\"}", "application/json", false).replace("\"status\":200", "\"status\":422");
        let record = parse(&text, &task()).unwrap();
        assert_eq!(record.replies[0].status, 422);
        let interim = answer(1, 1).replace("\"status\":200", "\"status\":100");
        assert!(parse(&interim, &task()).err().unwrap().contains("final HTTP status"));
    }

    #[test]
    fn recorded_fixtures_parse_the_same_twice() {
        let fixtures = [
            (include_str!("../../tests/agent-suite/tasks/markdown-spec-qa/task.txt"), include_str!("../../tests/fixtures/suite-replay/tasks/markdown-spec-qa/transcript.jsonl")),
            (include_str!("../../tests/agent-suite/tasks/refuse-outside-path/task.txt"), include_str!("../../tests/fixtures/suite-replay/tasks/refuse-outside-path/transcript.jsonl")),
            (include_str!("../../tests/agent-suite/tasks/structured-output/task.txt"), include_str!("../../tests/fixtures/suite-replay/tasks/structured-output/transcript.jsonl")),
        ];
        for (task, transcript) in fixtures {
            let task = task::parse(task).unwrap();
            let (a, b) = (parse(transcript, &task).unwrap(), parse(transcript, &task).unwrap());
            assert_eq!(a.calls, b.calls);
            let replies = |r: Recording| r.replies.into_iter().map(|r| (r.status, r.content_type, r.body)).collect::<Vec<_>>();
            assert_eq!(replies(a), replies(b));
        }
    }

    #[test]
    fn broken_recorded_fixture_has_a_clear_error() {
        let task = task::parse(include_str!("../../tests/agent-suite/tasks/structured-output/task.txt")).unwrap();
        let error = parse(include_str!("../../tests/fixtures/suite-replay/broken.jsonl"), &task).err().unwrap();
        assert!(error.starts_with("line 1:"), "{error}");
        assert!(error.contains("invalid record JSON"), "{error}");
    }
}
