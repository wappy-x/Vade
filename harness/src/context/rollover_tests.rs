use super::*;
use crate::agent::{ModelRequest, RunObserver};
use crate::agent_wire::AssistantTurn;
use crate::contract::InputCount;
use crate::json;

const RESERVE: usize = 256;
const SUMMARY: &str = "Plan: finish the parser.\nDecisions: keep std only.\nOpen tasks: tests.\nFiles: src/lib.rs edited.\nFacts: none.";

fn limits(window: usize) -> Limits {
    Limits { max_context_tokens: window, output_reserve_tokens: RESERVE, ..Limits::default() }
}

/// A deterministic count: one token per four request bytes.
fn exact(c: &Conversation, window: usize) -> io::Result<Usage> {
    let tokens = c.serialized_bytes() / 4;
    Usage::for_conversation(c).with_count(&InputCount::Exact {
        prompt_tokens: tokens as u64,
        output_tokens: RESERVE as u64,
        context_limit: window as u64,
        admitted: tokens + RESERVE <= window,
    })
}

/// The fake model host: exact counts and one fixed summary per request.
struct Fake {
    window: usize,
    requests: Vec<String>,
}
impl ContextHost for Fake {
    fn count(&mut self, c: &Conversation) -> io::Result<Usage> {
        exact(c, self.window)
    }
    fn summarize(&mut self, request: &Conversation) -> io::Result<String> {
        self.requests.push(request.request(Limits::default().wire).to_json()?);
        Ok(SUMMARY.into())
    }
}

/// `turns` user turns. Each turn reads lines, edits a file, runs a command
/// and answers. Tool outputs carry a marker that must never reach a summary.
fn chat(turns: usize) -> Conversation {
    let mut messages = vec![r#"{"role":"system","content":"stable rules"}"#.to_owned()];
    for i in 0..turns {
        let output = format!("TOOL-OUTPUT-{i} {}", "fn body() {} ".repeat(200));
        let patch = json::quote(&format!("*** Begin Patch\n*** Update File: src/lib.rs\n@@\n-old{i}\n+new{i}\n*** End Patch"));
        messages.push(format!(r#"{{"role":"user","content":"ask number {i}: change the parser"}}"#));
        messages.push(format!(r#"{{"role":"assistant","content":"","tool_calls":[{{"id":"r{i}","type":"function","function":{{"name":"workspace_read_lines","arguments":{}}}}},{{"id":"e{i}","type":"function","function":{{"name":"apply_patch","arguments":{}}}}},{{"id":"s{i}","type":"function","function":{{"name":"workspace_shell","arguments":{}}}}}]}}"#,
            json::quote(&format!(r#"{{"path":"src/lib.rs","start_line":{},"end_line":{}}}"#, 10 + i, 40 + i)),
            json::quote(&json::stringify(&Value::Object(vec![("patch".into(), json::parse(&patch).unwrap())])).unwrap()),
            json::quote(r#"{"argv":["cargo","test"],"timeout_ms":1000}"#)));
        messages.push(format!(r#"{{"role":"tool","tool_call_id":"r{i}","content":{}}}"#, json::quote(&output)));
        messages.push(r#"{"role":"tool","tool_call_id":"e{i}","content":"Applied 1 file operations."}"#.replace("{i}", &i.to_string()));
        messages.push(format!(r#"{{"role":"tool","tool_call_id":"s{i}","content":"status: Exited(Some(0))\nok {i}"}}"#));
        messages.push(format!(r#"{{"role":"assistant","content":"answer {i}: {}"}}"#, "done and explained. ".repeat(40)));
    }
    let request = format!(
        r#"{{"model":"fake","tools":[{{"type":"function","function":{{"name":"workspace_read_lines"}}}},{{"type":"function","function":{{"name":"apply_patch"}}}},{{"type":"function","function":{{"name":"workspace_shell"}}}}],"messages":[{}]}}"#,
        messages.join(",")
    );
    Conversation::from_request(json::parse(&request).unwrap(), Limits::default()).unwrap()
}

#[test]
fn budget_math_tracks_prompt_tokens_against_the_window() {
    let c = chat(2);
    let window = 10_000;
    let usage = exact(&c, window).unwrap();
    let budget = Budget::from_usage(1, &usage, 131_072, RESERVE);
    assert_eq!((budget.prompt, budget.window, budget.reserve, budget.exact), (c.serialized_bytes() / 4, window, RESERVE, true));
    assert_eq!(budget.left(), window - budget.prompt - RESERVE);
    assert!(budget.fits(budget.left()) && !budget.fits(budget.left() + 1));
    assert!(budget.line().starts_with(&format!("context after turn 1: {} of {window} prompt tokens (", budget.prompt)));
    // The reply reservation never exceeds a quarter of the window.
    let small = Budget { turn: 1, prompt: 0, window: 4096, reserve: reply_reservation(4096, 8192), exact: true };
    assert_eq!(small.reserve, 1024);
    // An estimate says so, and the tracker keeps the newest 64 turns.
    let estimate = Budget::from_usage(1, &Usage::for_conversation(&c), window, RESERVE);
    assert!(!estimate.exact && estimate.line().contains(" about "));
    let mut tracker = BudgetTracker::default();
    for prompt in 0..70 {
        tracker.record(prompt, window, RESERVE, true);
    }
    assert_eq!(tracker.turns().count(), 64);
    assert_eq!(tracker.last().map(|b| (b.turn, b.prompt)), Some((70, 69)));
    assert_eq!(tracker.turns().next().map(|b| b.turn), Some(7));
    assert!(tracker.new_source(5) && !tracker.new_source(5) && tracker.new_source(6), "one entry per response");
    assert_eq!(summary_cap(8192), 3072);
    assert_eq!((summary_cap(1024), summary_cap(1 << 20)), (1024, 8192));
}

#[test]
fn rollover_keeps_recent_turns_verbatim_and_the_prefix_exact() {
    let c = chat(5);
    let window = c.serialized_bytes() / 4;
    let before = exact(&c, window).unwrap();
    assert!(!before.fits(window, RESERVE), "the next turn must not fit");
    let mut host = Fake { window, requests: Vec::new() };
    let (plan, after) = rollover(&c, KEEP_TURNS, limits(window), &before, &mut host).unwrap();
    assert!(after.fits(window, RESERVE) && after.exact.unwrap().0 < before.exact.unwrap().0);
    assert_eq!(host.requests.len(), 1, "one summary request");
    let starts = turn_starts(&c);
    let cut = starts[starts.len() - KEEP_TURNS];
    let m = plan.conversation().messages();
    assert_eq!(m[0], c.messages()[0], "system prefix bytes unchanged");
    assert_eq!(&m[2..], &c.messages()[cut..], "kept turns are verbatim");
    assert!(is_summary(&m[1]));
    let block = m[1].value().get("content").and_then(Value::as_str).unwrap();
    assert!(block.contains(SUMMARY));
    assert!(block.contains("- src/lib.rs: read 10-40, 11-41, 12-42; edit calls 3, applied 3"), "{block}");
    assert!(block.contains("argv: cargo test -> status: Exited(Some(0))"));
    assert!(block.ends_with("(verbatim, oldest first):\n--- ask, 31 bytes ---\nask number 0: change the parser\n--- ask, 31 bytes ---\nask number 1: change the parser\n--- ask, 31 bytes ---\nask number 2: change the parser\n"), "{block}");
    assert!(!block.contains("TOOL-OUTPUT"), "tool outputs stay by reference");
    let (kept, tokens) = plan.rollover().unwrap();
    assert_eq!(kept, KEEP_TURNS);
    assert!(tokens > 0 && tokens < after.exact.unwrap().0);
    assert_eq!(plan.status_line().unwrap(), format!("[context compacted: kept 2 turns, summary {tokens} tokens]"));
    // The summary request keeps the frozen prefix and sends references only.
    let request = &host.requests[0];
    assert!(request.contains("stable rules") && request.contains("Write a compact state block"));
    assert!(request.contains("[tool call workspace_read_lines] src/lib.rs:10-40"));
    assert!(request.contains("[tool result workspace_read_lines] ") && !request.contains("TOOL-OUTPUT-0 fn"));
    assert!(!request.contains("ask number 3"), "kept turns are not summarized");
}

#[test]
fn kept_turns_shrink_to_fit_and_stop_when_nothing_fits() {
    let c = chat(4);
    let starts = turn_starts(&c);
    let floor = |kept: usize| exact(&bare(&c, starts[starts.len() - kept], limits(usize::MAX)).unwrap(), usize::MAX).unwrap().exact.unwrap().0;
    let block = |kept: usize, window| (state_block(&c, starts[starts.len() - kept], "").len() + summary_cap(window)).div_ceil(3) + 16;
    // Room for one kept turn and a full block, but not for two.
    let window = floor(1) + block(1, 4096) + RESERVE + 8;
    assert!(floor(2) + block(2, window) + RESERVE > window);
    let before = exact(&c, window).unwrap();
    let mut host = Fake { window, requests: Vec::new() };
    let (plan, _) = rollover(&c, KEEP_TURNS, limits(window), &before, &mut host).unwrap();
    assert_eq!(plan.rollover().unwrap().0, 1);
    assert_eq!(plan.conversation().messages()[2..], c.messages()[*starts.last().unwrap()..]);
    // Even one kept turn and a block do not fit: no model request, a hard stop.
    let window = floor(1) + RESERVE;
    let mut host = Fake { window, requests: Vec::new() };
    let before = exact(&c, window).unwrap();
    match rollover(&c, KEEP_TURNS, limits(window), &before, &mut host) {
        Err(RolloverError::NoRoom { needed, window: w }) => {
            assert_eq!(w, window);
            assert!(needed > window, "the hard stop counts the reply room: {needed}");
            assert!(hard_stop(needed, w).starts_with("This chat cannot continue: even after compaction"));
        }
        other => panic!("expected NoRoom, got {other:?}"),
    }
    assert!(host.requests.is_empty());
    // One turn has nothing older to summarize; an unknown outcome is kept exact.
    let one = chat(1);
    assert!(matches!(rollover_room(&one, KEEP_TURNS, limits(100), &exact(&one, 100).unwrap(), &mut host), Err(RolloverError::NotApplicable(_))));
    let text = json::stringify(&json::parse(&c.request(Limits::default().wire).to_json().unwrap()).unwrap()).unwrap()
        .replace("Applied 1 file operations.", r#"{\"error\":\"unknown_outcome\"}"#);
    let uncertain = Conversation::from_request(json::parse(&text).unwrap(), Limits::default()).unwrap();
    assert!(matches!(rollover_room(&uncertain, KEEP_TURNS, limits(window), &before, &mut host), Err(RolloverError::NotApplicable("an older tool outcome is unknown"))));
}

#[test]
fn a_count_only_host_cannot_summarize() {
    let c = chat(3);
    let window = c.serialized_bytes() / 4;
    let mut count = |c: &Conversation| exact(c, window);
    let error = rollover(&c, KEEP_TURNS, limits(window), &exact(&c, window).unwrap(), &mut CountOnly(&mut count, false)).unwrap_err();
    assert!(matches!(error, RolloverError::Failed(ref e) if e.kind() == io::ErrorKind::Unsupported), "{error}");
}

/// A fake model: exact counts, a summary for the summary request, and a
/// plain answer otherwise. It records whether the prompt was rewritten.
struct FakeModel {
    window: usize,
    rewrites: usize,
    prompts: Vec<String>,
    tool_call: bool,
}
impl Model for FakeModel {
    fn prompt_rewritten(&mut self) {
        self.rewrites += 1;
    }
    fn count_input(&mut self, request: ModelRequest<'_>, _: &AtomicBool) -> io::Result<InputCount> {
        let tokens = request.to_json()?.len() / 4;
        Ok(InputCount::Exact { prompt_tokens: tokens as u64, output_tokens: RESERVE as u64, context_limit: self.window as u64, admitted: tokens + RESERVE <= self.window })
    }
    fn complete(&mut self, request: ModelRequest<'_>, _: &AtomicBool, _: &mut dyn FnMut(&str) -> io::Result<()>) -> io::Result<AssistantTurn> {
        let body = request.to_json()?;
        self.prompts.push(body.clone());
        let message = if body.contains("Write a compact state block") {
            if self.tool_call {
                r#"{"role":"assistant","content":"","tool_calls":[{"id":"x","type":"function","function":{"name":"workspace_shell","arguments":"{}"}}]}"#.to_owned()
            } else {
                format!(r#"{{"role":"assistant","content":{}}}"#, json::quote(SUMMARY))
            }
        } else {
            r#"{"role":"assistant","content":"final answer"}"#.to_owned()
        };
        let response = format!(r#"{{"choices":[{{"message":{message},"finish_reason":"stop"}}]}}"#);
        AssistantTurn::from_response(&json::parse(&response)?, request.wire)
    }
}

#[test]
fn summarize_with_accepts_text_and_refuses_tool_calls() {
    let c = chat(3);
    let request = summary_request(&c, turn_starts(&c)[1], limits(1 << 20), 1 << 20).unwrap();
    let mut model = FakeModel { window: 1 << 20, rewrites: 0, prompts: Vec::new(), tool_call: false };
    assert_eq!(summarize_with(&mut model, &request, Limits::default().wire, &AtomicBool::new(false)).unwrap(), SUMMARY);
    assert_eq!(model.rewrites, 2, "the prompt is rewritten before and after");
    model.tool_call = true;
    assert!(summarize_with(&mut model, &request, Limits::default().wire, &AtomicBool::new(false)).is_err());
}

/// The trigger inside the agent loop: a journal-backed observer rolls over
/// when the next turn would not fit, and a resume replays the compaction.
#[test]
fn the_loop_rolls_over_when_the_next_turn_would_not_fit_and_the_journal_replays_it() {
    struct Rolling(crate::journal::Journal, Vec<String>);
    impl RunObserver for Rolling {
        fn checkpoint(&mut self, c: &Conversation, m: &[Message]) -> io::Result<()> {
            self.0.checkpoint(c, m)
        }
        fn before_hosted_request(&mut self, c: &mut Conversation, l: Limits, _: &AtomicBool, u: &Usage, host: &mut dyn ContextHost) -> io::Result<bool> {
            if u.fits(l.max_context_tokens, l.output_reserve_tokens) {
                return Ok(false);
            }
            let (plan, _) = rollover(c, KEEP_TURNS, l, u, host).map_err(|e| io::Error::other(e.to_string()))?;
            self.0.compact(&plan)?;
            self.1.push(plan.status_line().unwrap());
            *c = plan.into_conversation();
            Ok(true)
        }
    }
    let fixture = crate::test_workspace::Fixture::new();
    let workspace = fixture.workspace();
    let mut c = chat(5);
    c.push_user(json::parse(r#"{"role":"user","content":"ask number 5: finish"}"#).unwrap(), Limits::default()).unwrap();
    let window = c.serialized_bytes() / 4;
    let mut observer = Rolling(crate::journal::Journal::create(&workspace, "rollover", &c).unwrap(), Vec::new());
    let mut model = FakeModel { window, rewrites: 0, prompts: Vec::new(), tool_call: false };
    struct NoTools;
    impl crate::agent::ToolExecutor for NoTools {
        fn execute(&mut self, _: crate::agent::ToolInvocation<'_>, _: &AtomicBool) -> Result<String, crate::agent::ToolError> {
            Err(crate::agent::ToolError::UnknownTool)
        }
    }
    let report = crate::agent::run_observed(&mut c, &mut model, &mut NoTools, limits(window), &AtomicBool::new(false), &mut |_| (), &mut observer);
    assert_eq!(report.reason, crate::agent::StopReason::Finished(FinishReason::Stop));
    assert_eq!(observer.1.len(), 1, "exactly one rollover");
    assert!(observer.1[0].starts_with("[context compacted: kept 2 turns, summary "));
    assert_eq!(model.prompts.len(), 2, "one summary request, then the real turn");
    assert!(model.prompts[1].contains(json::quote(SUMMARY).trim_matches('"')) && model.prompts[1].contains("ask number 5: finish"));
    assert!(model.rewrites >= 3, "summary before/after and the rewritten chat");
    // The journal records what was summarized and the summary itself.
    let status = observer.0.compaction_status();
    assert!(status.contains("Model-written (kept 2 turns, ") && status.contains(SUMMARY), "{status}");
    drop(observer);
    let (_, resumed, _) = crate::journal::Journal::resume(&workspace, "rollover", Limits::default()).unwrap();
    assert_eq!(resumed.messages(), c.messages(), "resume replays the compaction and the answer");
    let raw = String::from_utf8_lossy(&std::fs::read(workspace.root().join(".vade/sessions/rollover.journal")).unwrap()).into_owned();
    for field in [r#""summary_origin":"model""#, r#""kept_turns":2"#, r#""removed_message_indices":[1,2,3"#] {
        assert!(raw.contains(field), "journal lacks {field}");
    }
}

#[test]
fn a_second_rollover_carries_the_earlier_asks_and_folds_the_earlier_summary() {
    let c = chat(5);
    let window = c.serialized_bytes() / 4;
    let mut host = Fake { window, requests: Vec::new() };
    let first = rollover(&c, KEEP_TURNS, limits(window), &exact(&c, window).unwrap(), &mut host).unwrap().0.into_conversation();
    // Three more turns of the same size, then a second rollover.
    let more = chat(8);
    let starts = turn_starts(&more);
    let mut messages = first.messages().to_vec();
    messages.extend_from_slice(&more.messages()[starts[5]..]);
    let grown = first.with_messages(messages, Limits::default()).unwrap();
    let second = rollover(&grown, KEEP_TURNS, limits(window), &exact(&grown, window).unwrap(), &mut host).unwrap().0;
    let m = second.conversation().messages();
    assert_eq!(m.iter().filter(|m| is_summary(m)).count(), 1, "the earlier block is folded, not stacked");
    let block = m[1].value().get("content").and_then(Value::as_str).unwrap();
    let expected: Vec<_> = (0..6).map(|i| format!("--- ask, 31 bytes ---\nask number {i}: change the parser\n")).collect();
    assert!(block.ends_with(&expected.concat()), "{block}");
    // The earlier block reached the model whole, in the digest.
    assert!(host.requests[1].contains("[earlier summary]") && host.requests[1].contains("Plan: finish the parser."));
    // A long history keeps the newest asks within 4 KiB and says what it left out.
    let long: Vec<_> = (0..40).map(|i| format!("ask {i}: {}", "word ".repeat(40))).collect();
    let mut request = String::from(r#"{"model":"fake","messages":[{"role":"system","content":"s"}"#);
    for ask in &long {
        request.push_str(&format!(r#",{{"role":"user","content":"{ask}"}},{{"role":"assistant","content":"ok"}}"#));
    }
    request.push_str(r#",{"role":"user","content":"now"}]}"#);
    let c = Conversation::from_request(json::parse(&request).unwrap(), Limits::default()).unwrap();
    let section = asks(&c, 1, c.messages().len() - 1);
    assert!(section.starts_with("[23 older asks omitted; the summary covers them]\n"), "{section}");
    assert!(section.len() <= 4096 + 64 && section.ends_with(&format!("{}\n", long[39])));
}

#[test]
fn an_unknown_outcome_in_a_kept_turn_does_not_block_the_rollover() {
    let c = chat(4);
    let text = c.request(Limits::default().wire).to_json().unwrap()
        .replace(r#""tool_call_id":"e3","content":"Applied 1 file operations.""#, r#""tool_call_id":"e3","content":"{\"error\":\"unknown_outcome\"}""#);
    let c = Conversation::from_request(json::parse(&text).unwrap(), Limits::default()).unwrap();
    assert!(has_unknown_outcome(&c));
    let window = c.serialized_bytes() / 4;
    let mut host = Fake { window, requests: Vec::new() };
    let (plan, _) = rollover(&c, KEEP_TURNS, limits(window), &exact(&c, window).unwrap(), &mut host).unwrap();
    let starts = turn_starts(&c);
    assert_eq!(plan.conversation().messages()[2..], c.messages()[starts[2]..], "the uncertain receipt stays exact");
}

#[test]
fn an_ask_that_quotes_the_section_headers_is_carried_whole() {
    let tricky = format!("paste: {ASKS}\n--- ask, 3 bytes ---\nabc\nend");
    let request = format!(r#"{{"model":"fake","messages":[{{"role":"system","content":"s"}},{{"role":"user","content":{}}},{{"role":"assistant","content":"ok"}},{{"role":"user","content":"now"}}]}}"#, json::quote(&tricky));
    let c = Conversation::from_request(json::parse(&request).unwrap(), Limits::default()).unwrap();
    let section = asks(&c, 1, 3);
    // The model's part quotes a header too; only the real section parses whole.
    let block = format!("Plan: quote {ASKS}\n--- ask, 1 bytes ---\nx\n\n{ASKS}\n{section}");
    let header = format!("{ASKS}\n");
    let found = block.match_indices(&header).find_map(|(at, _)| carried(&block[at + header.len()..]));
    assert_eq!(found, Some((0, vec![tricky])));
    assert_eq!(carried("(none)"), Some((0, Vec::new())));
    // An earlier block that left asks out: a later block still counts them.
    let block = format!("Plan: p\n\n{ASKS}\n[5 older asks omitted; the summary covers them]\n--- ask, 1 bytes ---\na\n");
    let request = format!(r#"{{"model":"fake","messages":[{{"role":"system","content":"s"}},{{"role":"user","name":"vade_context_summary","content":{}}},{{"role":"user","content":"b"}},{{"role":"assistant","content":"ok"}},{{"role":"user","content":"now"}}]}}"#, json::quote(&block));
    let c = Conversation::from_request(json::parse(&request).unwrap(), Limits::default()).unwrap();
    assert_eq!(asks(&c, 1, 4), "[5 older asks omitted; the summary covers them]\n--- ask, 1 bytes ---\na\n--- ask, 1 bytes ---\nb\n");
    assert_eq!(carried("--- ask, 9 bytes ---\nshort\n"), None);
}

#[test]
fn a_captured_command_result_keeps_its_status_line_by_reference() {
    // Output over 4 KiB is captured: a notice line, then a JSON envelope.
    let output = format!("status: Exited(Some(0))\n{}", "TOOL-OUTPUT ".repeat(600));
    let captured = crate::artifacts::excerpt(&output, "a-1", 8192).unwrap();
    assert!(captured.starts_with("Captured tool result excerpt"));
    assert_eq!(first_line(&captured), "status: Exited(Some(0))");
    assert_eq!(result_reference(&captured), format!("{} bytes; status: Exited(Some(0))", captured.len()));
    // A later chunk does not hold the header: no status is quoted.
    let later = format!("Captured tool result excerpt; notice\n{}", crate::artifacts::chunk(&output, "a-1", 100, 4096, 8192).unwrap());
    assert_eq!(first_line(&later), "");
    let call = json::quote(r#"{"argv":["cargo","test"]}"#);
    let request = format!(r#"{{"model":"fake","messages":[{{"role":"system","content":"s"}},{{"role":"user","content":"run the tests"}},{{"role":"assistant","content":"","tool_calls":[{{"id":"s0","type":"function","function":{{"name":"workspace_shell","arguments":{call}}}}}]}},{{"role":"tool","tool_call_id":"s0","content":{}}},{{"role":"assistant","content":"ok"}},{{"role":"user","content":"now"}}]}}"#, json::quote(&captured));
    let c = Conversation::from_request(json::parse(&request).unwrap(), Limits::default()).unwrap();
    let block = state_block(&c, 5, "");
    assert!(block.contains("- argv: cargo test -> status: Exited(Some(0))\n"), "{block}");
    assert!(!block.contains("TOOL-OUTPUT") && !block.contains("Captured tool result"), "{block}");
}

/// Counts as an engine gives them for an automatic reply (contract.rs
/// `admit_adjusted`): the reply takes the room the window has left, at most
/// 16384 tokens and 256 below the window.
struct AutoReply {
    window: usize,
}
impl ContextHost for AutoReply {
    fn count(&mut self, c: &Conversation) -> io::Result<Usage> {
        let tokens = c.serialized_bytes() / 4;
        let output = 16_384.min(self.window.saturating_sub(tokens).saturating_sub(256));
        Usage::for_conversation(c).with_count(&InputCount::Exact {
            prompt_tokens: tokens as u64,
            output_tokens: output as u64,
            context_limit: self.window as u64,
            admitted: output > 0,
        })
    }
    fn summarize(&mut self, _: &Conversation) -> io::Result<String> {
        Ok(SUMMARY.into())
    }
    fn automatic_output(&self) -> bool {
        true
    }
}

#[test]
fn a_byte_read_names_its_range_and_a_search_does_not() {
    let read = |args: &str| call_paths("workspace_read", args);
    assert_eq!(read(r#"{"path":"a.rs","limit":256}"#), vec![("a.rs".to_owned(), Some("bytes 0+256".to_owned()))]);
    assert_eq!(read(r#"{"path":"a.rs","offset":64}"#), vec![("a.rs".to_owned(), Some("bytes from 64".to_owned()))]);
    assert_eq!(read(r#"{"path":"a.rs","offset":64,"limit":16}"#), vec![("a.rs".to_owned(), Some("bytes 64+16".to_owned()))]);
    assert_eq!(read(r#"{"path":"a.rs"}"#), vec![("a.rs".to_owned(), Some("bytes from 0".to_owned()))]);
    // Search paging uses offset and limit too; they are not bytes of the file.
    assert_eq!(call_paths("workspace_search", r#"{"path":"src","literal":"x","offset":0,"limit":16}"#), vec![("src".to_owned(), None)]);
}

#[test]
fn the_reply_room_follows_the_output_mode_not_the_count() {
    let c = chat(1);
    let count = |prompt: usize, output: usize, window: usize| Usage::for_conversation(&c).with_count(&InputCount::Exact {
        prompt_tokens: prompt as u64,
        output_tokens: output as u64,
        context_limit: window as u64,
        admitted: true,
    }).unwrap();
    // Automatic: a 114,000-token floor whose count chose the full 16,384-token
    // reply. With a 3,000-token block the reply shrinks to 13,816, so only the
    // 8,192-token reservation and 256 tokens are needed, as for a larger floor.
    let automatic = Limits { max_context_tokens: 131_072, output_reserve_tokens: 8192, ..Limits::default() };
    assert_eq!(needed_with(&count(114_000, 16_384, 131_072), 3000, 131_072, automatic, true), 125_448);
    assert_eq!(needed_with(&count(115_000, 15_816, 131_072), 3000, 131_072, automatic, true), 126_448);
    // Explicit 20,000-token reply whose count fills the window exactly: the
    // reply keeps all of its room.
    let explicit = Limits { max_context_tokens: 32_768, output_reserve_tokens: 20_000, ..Limits::default() };
    assert_eq!(needed_tokens(&count(12_512, 20_000, 32_768), 32_768, explicit, false), 32_512);
}

#[test]
fn an_automatic_reply_count_still_leaves_room_for_the_block() {
    // A small window: the automatic reply fills what is left, so the old check
    // (count + block + that reply) never fit and the rollover never ran.
    let c = chat(4);
    let window = 16_384;
    let mut host = AutoReply { window };
    let before = host.count(&c).unwrap();
    assert!(input(&before) < window && before.exact.unwrap().1 < 16_384, "the window clamps the reply");
    let (plan, after) = rollover(&c, KEEP_TURNS, limits(window), &before, &mut host).unwrap();
    assert!(plan.rollover().is_some() && input(&after) < input(&before));
    assert!(after.fits(window, RESERVE));
}

/// Exact counts with an explicit reply (`--max-output-tokens`): the engine
/// counts that reply as asked, whatever the window leaves.
struct ExplicitReply {
    window: usize,
    output: usize,
    summaries: usize,
}
impl ContextHost for ExplicitReply {
    fn count(&mut self, c: &Conversation) -> io::Result<Usage> {
        let tokens = c.serialized_bytes() / 4;
        Usage::for_conversation(c).with_count(&InputCount::Exact {
            prompt_tokens: tokens as u64,
            output_tokens: self.output as u64,
            context_limit: self.window as u64,
            admitted: tokens + self.output <= self.window,
        })
    }
    fn summarize(&mut self, _: &Conversation) -> io::Result<String> {
        self.summaries += 1;
        Ok(SUMMARY.into())
    }
}

#[test]
fn an_explicit_reply_keeps_its_full_room_in_the_kept_turns_rule() {
    // Two kept turns fit with the 256-token reserve but not with the asked
    // 24,000-token reply; one kept turn fits with it. At these windows the
    // block cap is its 8 KiB maximum.
    let c = chat(4);
    let starts = turn_starts(&c);
    let output = 24_000;
    let floor = |kept: usize| (bare(&c, starts[starts.len() - kept], limits(usize::MAX)).unwrap().serialized_bytes() / 4)
        + (state_block(&c, starts[starts.len() - kept], "").len() + 8192).div_ceil(3) + 16;
    let window = (floor(1) + output).max(floor(2) + 256);
    assert!(floor(2) + output > window && summary_cap(window) == 8192);
    let mut host = ExplicitReply { window, output, summaries: 0 };
    let before = host.count(&c).unwrap();
    let (plan, after) = rollover(&c, KEEP_TURNS, limits(window), &before, &mut host).unwrap();
    assert_eq!(plan.rollover().unwrap().0, 1, "the reply room leaves one kept turn");
    assert_eq!(host.summaries, 1, "the rule picks one turn before the model writes");
    assert!(after.fits(window, RESERVE));
    // The hard stop names the full reply room: more than the window.
    assert!(needed_tokens(&before, window, limits(window), false) > window);
}

/// A host with no exact count: every count is the local estimate.
struct Estimate {
    summaries: usize,
}
impl ContextHost for Estimate {
    fn count(&mut self, c: &Conversation) -> io::Result<Usage> {
        Ok(Usage::for_conversation(c))
    }
    fn summarize(&mut self, _: &Conversation) -> io::Result<String> {
        self.summaries += 1;
        Ok(SUMMARY.into())
    }
}

#[test]
fn an_estimate_keeps_the_full_reserve_in_the_kept_turns_rule() {
    // A small window caps the protected reservation at a quarter of the window,
    // but an estimate fits only with the full reserve: find a window where two
    // kept turns fit with the quarter and only one fits with the reserve.
    let c = chat(4);
    let starts = turn_starts(&c);
    let reserve = 8192;
    let with_reserve = |window: usize| Limits { max_context_tokens: window, output_reserve_tokens: reserve, ..Limits::default() };
    let floor = |kept: usize, window: usize| Usage::for_conversation(&bare(&c, starts[starts.len() - kept], limits(usize::MAX)).unwrap()).estimated
        + (state_block(&c, starts[starts.len() - kept], "").len() + summary_cap(window)).div_ceil(3) + 16;
    let window = (4096..32_768).find(|&w| floor(1, w) + reserve <= w && floor(2, w) + reserve > w && floor(2, w) + reply_reservation(w, reserve) <= w)
        .expect("a window between the two rules");
    let mut host = Estimate { summaries: 0 };
    let before = host.count(&c).unwrap();
    let (plan, after) = rollover(&c, KEEP_TURNS, with_reserve(window), &before, &mut host).unwrap();
    assert_eq!(plan.rollover().unwrap().0, 1, "the full reserve leaves one kept turn");
    assert_eq!(host.summaries, 1, "the rule picks one turn before the model writes");
    assert!(after.fits(window, reserve));
}

#[test]
fn the_resume_note_is_not_an_ask() {
    // A resumed chat carries the harness note as a user message between turns.
    let c = chat(3);
    let mut request = json::parse(&c.request(Limits::default().wire).to_json().unwrap()).unwrap();
    let Value::Object(fields) = &mut request else { unreachable!() };
    let Some((_, Value::Array(messages))) = fields.iter_mut().find(|(k, _)| k == "messages") else { unreachable!() };
    let at = messages.iter().position(|m| m.get("content").and_then(Value::as_str) == Some("ask number 1: change the parser")).unwrap();
    messages.insert(at, json::parse(&format!(r#"{{"role":"user","content":{}}}"#, json::quote(RESUME_NOTE))).unwrap());
    let c = Conversation::from_request(request, Limits::default()).unwrap();
    let starts = turn_starts(&c);
    let block = state_block(&c, starts[starts.len() - 1], "");
    assert!(block.contains("ask number 0") && block.contains("ask number 1"));
    assert!(!block.contains("Transcript resumed"), "{block}");
}

/// `chat(turns)` with a harness recovery prompt and the automatic check
/// (call and result) inside the first turn.
fn chat_with_pins(turns: usize) -> Conversation {
    let c = chat(turns);
    let parse = |text: &str| Message::from_value(json::parse(text).unwrap(), Limits::default().wire).unwrap();
    let mut messages = c.messages().to_vec();
    let at = turn_starts(&c)[1];
    messages.splice(at..at, [
        parse(r#"{"role":"developer","content":"The tool call apply_patch src/lib.rs (untrusted tool/path text) was discarded because it exceeded the 4096-token reply limit."}"#),
        parse(r#"{"role":"assistant","content":"","tool_calls":[{"id":"vade-check-9","type":"function","function":{"name":"workspace_shell","arguments":"{\"argv\":[\"make\",\"check\"],\"timeout_ms\":1000}"}}]}"#),
        parse(r#"{"role":"tool","tool_call_id":"vade-check-9","content":"status: Exited(Some(1))\nerror: check failed"}"#),
    ]);
    c.with_messages(messages, Limits::default()).unwrap()
}

#[test]
fn harness_prompts_and_the_latest_check_survive_rollover_and_pressure_plans() {
    let c = chat_with_pins(4);
    let pinned: Vec<String> = c.messages().iter()
        .filter(|m| m.role() == "developer" || m.tool_call_id() == Some("vade-check-9") || m.tool_calls().iter().any(|call| call.id == "vade-check-9"))
        .map(|m| m.serialized_json().to_owned()).collect();
    assert_eq!(pinned.len(), 3);
    let starts = turn_starts(&c);
    let cut = starts[starts.len() - 1];
    let plan = rollover_plan(&c, cut, 1, SUMMARY, limits(1 << 20)).unwrap();
    let after = plan.conversation().messages();
    // Prefix, state block, the pinned messages in order, then the kept turn.
    assert_eq!(after[0].serialized_json(), c.messages()[0].serialized_json());
    assert_eq!(after[1].value().get("name").and_then(Value::as_str), Some(SUMMARY_NAME));
    assert_eq!(after[2..5].iter().map(|m| m.serialized_json()).collect::<Vec<_>>(), pinned);
    assert_eq!(after[5..].iter().map(|m| m.serialized_json()).collect::<Vec<_>>(),
        c.messages()[cut..].iter().map(|m| m.serialized_json()).collect::<Vec<_>>());
    assert!(!plan.removed_indices.iter().any(|&i| c.messages()[i].role() == "developer"));
    // The floor that the kept-turns rule counts holds the pinned bytes too.
    let floor = bare(&c, cut, limits(1 << 20)).unwrap();
    assert_eq!(floor.messages()[1..4].iter().map(|m| m.serialized_json()).collect::<Vec<_>>(), pinned);
    // Pressure keeps only the newest rounds, but still the check and the prompt.
    let plan = pressure_plan(&c, 1, 12 * 1024, limits(1 << 20)).unwrap();
    let kept: Vec<_> = plan.conversation().messages().iter().map(|m| m.serialized_json()).collect();
    assert!(pinned.iter().all(|p| kept.contains(&p.as_str())));
}
