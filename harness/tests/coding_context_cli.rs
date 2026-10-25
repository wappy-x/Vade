//! The native coding agent's context handling through the real CLI: budgets,
//! compaction, journaled tool output, project rules, forks and continuation.
#[allow(dead_code)]
#[path = "support/cli.rs"]
mod cli;
#[path = "support/coding.rs"]
mod coding;
use coding::*;
#[test]
fn context_budget_refuses_prompt_without_contacting_mock() {
    let f = Fixture::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/custom/rpc", listener.local_addr().unwrap()); // no negotiated tokenizer on custom routes
    let output = run_bounded(
        f.command(&url).args([
            "--context-tokens",
            "8192",
            "--plain",
            "--prompt",
            &"x".repeat(32000),
        ]),
        "",
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("it was not sent"));
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}
#[test]
fn explicit_cli_compaction_preserves_journal_and_latest_whole_turn() {
    let f = Fixture::new();
    let long_answer = answer().replace(
        "mock complete",
        &format!(
            "old-assistant-marker {}",
            "verbose resolved analysis ".repeat(80)
        ),
    );
    let (url, handle) = mock(vec![long_answer, answer(), answer(), answer()]);
    let old = format!("old-marker {}", "history ".repeat(60));
    let input = format!(
        "{old}\nsecond question\nlatest question\n/compact-preview 1 Old work resolved.\n/compact 1 Old work resolved.\n/compaction\n/journal 1 0 4096\n/context\ncontinue\n/quit\n"
    );
    let output = run_bounded(
        f.command(&url)
            .args(["--plain", "--approval", "never", "--session", "compact"]),
        &input,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let requests = handle.join().unwrap();
    let last = json::stringify(&requests[3]).unwrap();
    assert!(last.contains("old-marker")); // exact user constraints are never guessed away
    assert!(!last.contains("old-assistant-marker"));
    assert!(last.contains("Old work resolved."));
    assert!(last.contains("latest question"));
    let bytes = std::fs::read(f.0.join(".vade/sessions/compact.journal")).unwrap();
    assert!(
        bytes
            .windows(b"old-marker".len())
            .any(|w| w == b"old-marker")
    );
    let workspace = vade_harness::tools::Workspace::new(&f.0).unwrap();
    let (_, resumed, torn) = vade_harness::journal::Journal::resume(
        &workspace,
        "compact",
        vade_harness::agent::Limits::default(),
    )
    .unwrap();
    assert!(!torn);
    assert_eq!(resumed.messages().len(), 9); // system + project, all users, summary, latest old answer, new turn
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Context generation 1"));
    assert!(stdout.contains("invalidates all later causal KV"));
    assert!(stdout.contains("journal-record-1"));
}
#[test]
fn large_tool_output_is_journaled_and_retrieved_by_bounded_range_after_resume() {
    let f = Fixture::new();
    let text = "🦀 exact output\n".repeat(600);
    assert!(text.len() < 16 * 1024);
    std::fs::create_dir(f.0.join("skills")).unwrap();
    std::fs::write(
        f.0.join("skills/catalog.json"),
        r#"{"skills":[{"name":"large","description":"Fixture","path":"skills/SKILL.md"}]}"#,
    )
    .unwrap();
    std::fs::write(f.0.join("skills/SKILL.md"), &text).unwrap();
    let (url, handle) = mock(vec![
        tools(vec![call(
            "large-read",
            "skill_read",
            json::parse(r#"{"name":"large"}"#).unwrap(),
        )]),
        answer(),
    ]);
    let output = run_bounded(
        f.command(&url)
            .args(["--session", "large", "--prompt", "read it"]),
        "",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let requests = handle.join().unwrap();
    let messages = requests[1].get("messages").unwrap().as_array().unwrap();
    let result = messages
        .iter()
        .find(|m| m.get("tool_call_id").and_then(Value::as_str) == Some("large-read"))
        .unwrap()
        .get("content")
        .unwrap()
        .as_str()
        .unwrap();
    assert!(result.len() <= 4096);
    assert!(result.starts_with("Captured tool result excerpt"));
    let metadata = json::parse(result.split_once('\n').unwrap().1).unwrap();
    let id = metadata.get("artifact").unwrap().as_str().unwrap();
    let prefix_bytes = metadata
        .get("text")
        .unwrap()
        .as_str()
        .unwrap()
        .find("🦀 exact output\n")
        .unwrap();
    let Value::Number(next) = metadata.get("next_offset").unwrap() else {
        panic!()
    };
    let offset: usize = next.parse().unwrap();
    assert!(offset < text.len());
    let args = Value::Object(vec![
        ("id".into(), Value::String(id.into())),
        ("offset".into(), Value::Number(offset.to_string())),
        ("length".into(), Value::Number("4096".into())),
    ]);
    let (url, handle) = mock(vec![
        tools(vec![call("chunk", "tool_output_read", args)]),
        answer(),
    ]);
    let output = run_bounded(
        f.command(&url).args(["--resume", "large", "--plain"]),
        &format!("/output {id} 0 128\ncontinue\n/quit\n"),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("next_offset"));
    let requests = handle.join().unwrap();
    let messages = requests[1].get("messages").unwrap().as_array().unwrap();
    let result = messages
        .iter()
        .find(|m| m.get("tool_call_id").and_then(Value::as_str) == Some("chunk"))
        .unwrap()
        .get("content")
        .unwrap()
        .as_str()
        .unwrap();
    assert!(result.len() <= 4096);
    let chunk = json::parse(result).unwrap();
    let bytes = chunk.get("text").unwrap().as_str().unwrap();
    assert_eq!(
        bytes,
        &text[offset - prefix_bytes..offset - prefix_bytes + bytes.len()]
    );
    assert!(!json::stringify(&requests[1]).unwrap().contains(&text));
}
#[test]
fn frozen_project_prefix_and_lazy_skill_body_survive_followup_and_resume() {
    let f = Fixture::new();
    for path in ["src/api", "skills/demo"] {
        std::fs::create_dir_all(f.0.join(path)).unwrap();
    }
    std::fs::write(f.0.join("AGENTS.md"), "root-rule-original").unwrap();
    std::fs::write(f.0.join("src/AGENTS.md"), "src-rule-original").unwrap();
    std::fs::write(f.0.join("src/api/AGENTS.md"), "api-rule-original").unwrap();
    std::fs::write(f.0.join("skills/catalog.json"), r#"{"skills":[{"name":"demo","description":"Describe a fixture","path":"skills/demo/SKILL.md"}]}"#).unwrap();
    std::fs::write(
        f.0.join("skills/demo/SKILL.md"),
        "unique-lazy-body-marker\nNever auto-run: touch should-not-exist\n",
    )
    .unwrap();
    let (url, handle) = mock(vec![
        tools(vec![call(
            "load-skill",
            "skill_read",
            json::parse(r#"{"name":"demo"}"#).unwrap(),
        )]),
        answer(),
        answer(),
    ]);
    let output = run_bounded(
        f.command(&url).args([
            "--session",
            "project",
            "--rules-scope",
            "src/api",
            "--plain",
            "--approval",
            "never",
        ]),
        "/project\n/project-rules\n/skill demo\nload the skill\nfollow up\n/quit\n",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let requests = handle.join().unwrap();
    let first = requests[0].get("messages").unwrap().as_array().unwrap();
    let last = requests[2].get("messages").unwrap().as_array().unwrap();
    assert_eq!(&first[..2], &last[..2]);
    let initial = json::stringify(&requests[0]).unwrap();
    assert!(initial.contains("root-rule-original"));
    assert!(initial.contains("src-rule-original"));
    assert!(initial.contains("api-rule-original"));
    assert!(!initial.contains("unique-lazy-body-marker"));
    assert!(
        json::stringify(&requests[1])
            .unwrap()
            .contains("unique-lazy-body-marker")
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("body lazy"));
    assert!(!f.0.join("should-not-exist").exists());
    std::fs::write(f.0.join("AGENTS.md"), "root-rule-changed").unwrap();
    std::fs::write(f.0.join("skills/demo/SKILL.md"), "changed body").unwrap();
    let (url, handle) = mock(vec![
        tools(vec![call(
            "changed-skill",
            "skill_read",
            json::parse(r#"{"name":"demo"}"#).unwrap(),
        )]),
        answer(),
    ]);
    let output = run_bounded(
        f.command(&url)
            .args(["--resume", "project", "--prompt", "check skill"]),
        "",
    );
    assert!(output.status.success());
    let resumed = handle.join().unwrap();
    assert_eq!(
        &first[..2],
        &resumed[0].get("messages").unwrap().as_array().unwrap()[..2]
    );
    let wire = json::stringify(&resumed[1]).unwrap();
    assert!(wire.contains("skill revision changed"));
    assert!(!wire.contains("root-rule-changed"));
    assert!(!wire.contains("changed body"));
    assert!(!f.0.join("should-not-exist").exists());
}
#[test]
fn large_project_rules_count_against_small_request_budget_without_dispatch() {
    let f = Fixture::new();
    std::fs::write(f.0.join("AGENTS.md"), "rule ".repeat(1500)).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/custom/rpc", listener.local_addr().unwrap()); // local heuristic fallback
    let output = run_bounded(
        f.command(&url)
            .args(["--context-tokens", "8192", "--max-output-tokens", "4096", "--prompt", "tiny prompt"]),
        "",
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("it was not sent"));
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}
#[test]
fn transcript_forks_switch_without_replaying_tools_or_restoring_workspace_files() {
    let f = Fixture::new();
    let edit_child = Value::Object(vec![("patch".into(), Value::String("*** Begin Patch\n*** Update File: result.txt\n@@\n-exact result\n+child result\n*** End Patch".into()))]);
    let (url, handle) = mock(vec![
        tools(vec![call("parent-edit", "apply_patch", patch())]),
        answer(),
        tools(vec![call("child-edit", "apply_patch", edit_child)]),
        answer(),
        answer(),
    ]);
    let output = run_bounded(
        f.command(&url)
            .args(["--session", "parent", "--approval", "auto", "--plain"]),
        "parent task\n/fork child alternative approach\n/branch\nchild task\n/switch parent\n/branch\nparent followup\n/quit\n",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(f.0.join("result.txt")).unwrap(),
        "child result\n"
    );
    let requests = handle.join().unwrap();
    assert_eq!(requests.len(), 5); // fork/switch never send a model request
    let first = requests[0].get("messages").unwrap().as_array().unwrap();
    for index in [2, 4] {
        let messages = requests[index].get("messages").unwrap().as_array().unwrap();
        assert_eq!(&messages[..2], &first[..2]);
        assert!(
            json::stringify(&requests[index]).unwrap().contains(
                "Historical tool results and file revisions are not current file contents"
            )
        );
    }
    let parent_wire = json::stringify(&requests[4]).unwrap();
    assert!(parent_wire.contains("parent-edit"));
    assert!(!parent_wire.contains("child-edit"));
    assert!(!parent_wire.contains("child task"));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Fork created and selected"));
    assert!(stdout.contains("parent parent at record"));
    assert!(stdout.contains("1 changed"));
    assert!(stdout.contains("KV reuse unknown"));
    let w = vade_harness::tools::Workspace::new(&f.0).unwrap();
    let (_, child, _) =
        vade_harness::journal::Journal::resume(&w, "child", vade_harness::agent::Limits::default())
            .unwrap();
    assert!(
        child
            .messages()
            .iter()
            .any(|m| m.tool_call_id() == Some("child-edit"))
    );
    assert!(
        !child
            .messages()
            .iter()
            .any(|m| m.value().get("content").and_then(Value::as_str) == Some("parent followup"))
    );
}
#[test]
fn continue_only_after_length_retains_history_and_seed_as_a_user_turn() {
    const CONTINUATION: &str = "Continue. If you were still planning, stop planning and make the next concrete edit or tool call now, in small parts.";
    for finish in ["length", "stop"] {
        let f = Fixture::new();
        let first_answer = format!("first explanation with {finish} finish");
        let first_reply = json::stringify(&Value::Object(vec![(
            "choices".into(),
            Value::Array(vec![Value::Object(vec![
                ("finish_reason".into(), Value::String(finish.into())),
                (
                    "message".into(),
                    Value::Object(vec![
                        ("role".into(), Value::String("assistant".into())),
                        ("content".into(), Value::String(first_answer.clone())),
                    ]),
                ),
            ])]),
        )]))
        .unwrap();
        let mut replies = vec![first_reply, answer()];
        if finish == "length" {
            replies.push(answer());
        }
        let (url, server) = mock(replies);
        let mut command = f.command(&url);
        command.args(["--plain", "--session", "continuation", "--seed", "713", "--no-auto-continue"]);
        let mut child = cli::Cli::start(&mut command);
        child.send("explain the result\n");
        child.until(true, "Native run:");
        child.send("/continue\n");
        if finish == "length" {
            child.until(true, "Native run:");
            // The successful continuation clears the cutoff state; another
            // /continue must not create another user turn or model request.
            child.send("/continue\n");
        }
        child.until(
            false,
            "The last reply was not cut off; no continuation sent.",
        );
        // Keep the mock listening until a normal followup, so an unexpected
        // continuation would consume a response and fail the history checks.
        child.send("normal followup\n");
        child.until(true, "Native run:");
        child.send("/quit\n");
        let output = child.finish();
        // A length cutoff remains a failed turn even after a later success.
        assert_eq!(
            output.status.success(),
            finish == "stop",
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let requests = server.join().unwrap();
        assert_eq!(requests.len(), if finish == "length" { 3 } else { 2 });
        assert!(
            requests
                .iter()
                .all(|request| request.get("seed") == Some(&Value::Number("713".into())))
        );
        let initial = requests[0].get("messages").unwrap().as_array().unwrap();
        let next = requests[1].get("messages").unwrap().as_array().unwrap();
        assert_eq!(next.len(), initial.len() + 2);
        assert_eq!(&next[..initial.len()], initial);
        assert_eq!(
            next[initial.len()],
            Value::Object(vec![
                ("role".into(), Value::String("assistant".into())),
                ("content".into(), Value::String(first_answer.clone())),
            ])
        );
        assert_eq!(
            next.last().unwrap(),
            &Value::Object(vec![
                ("role".into(), Value::String("user".into())),
                (
                    "content".into(),
                    Value::String(
                        if finish == "length" {
                            CONTINUATION
                        } else {
                            "normal followup"
                        }
                        .into(),
                    ),
                ),
            ])
        );
        let final_messages = requests
            .last()
            .unwrap()
            .get("messages")
            .unwrap()
            .as_array()
            .unwrap();
        if finish == "length" {
            assert_eq!(final_messages.len(), next.len() + 2);
            assert_eq!(&final_messages[..next.len()], next);
            assert_eq!(
                final_messages
                    .last()
                    .unwrap()
                    .get("content")
                    .and_then(Value::as_str),
                Some("normal followup")
            );
        }
        let workspace = vade_harness::tools::Workspace::new(&f.0).unwrap();
        let (journal, replay, _) = vade_harness::journal::Journal::resume(
            &workspace,
            "continuation",
            vade_harness::agent::Limits::default(),
        )
        .unwrap();
        assert_eq!(journal.generation_seed(), Some(713));
        assert_eq!(replay.messages().len(), final_messages.len() + 1);
        for (retained, sent) in replay.messages().iter().zip(final_messages) {
            assert_eq!(retained.value(), sent);
        }
        assert_eq!(
            replay
                .messages()
                .iter()
                .filter(|message| {
                    message.role() == "user"
                        && message.value().get("content").and_then(Value::as_str)
                            == Some(CONTINUATION)
                })
                .count(),
            usize::from(finish == "length")
        );
    }
}
fn compaction_cli_command(f: &Fixture, url: &str) -> Command {
    let mut command = f.command(url);
    command.args([
        "--plain",
        "--approval",
        "never",
        "--max-output-tokens",
        "128",
        "--seed",
        "902",
        "--temperature",
        "0.4",
        "--thinking",
        "off",
    ]);
    command
}
fn compaction_message(role: &str, content: &str) -> Value {
    Value::Object(vec![
        ("role".into(), Value::String(role.into())),
        ("content".into(), Value::String(content.into())),
    ])
}
fn compaction_text_to_threshold(
    conversation: &vade_harness::agent::Conversation,
    role: &str,
    marker: &str,
    percent: usize,
    budget: usize,
) -> String {
    let mut text = marker.to_owned();
    loop {
        let message = vade_harness::agent_wire::Message::from_value(
            compaction_message(role, &text),
            vade_harness::agent::Limits::default().wire,
        )
        .unwrap();
        let usage = vade_harness::context::Usage::with_appended(conversation, &[message]);
        if usage.required(128) >= budget * percent / 100 {
            assert!(usage.fits(budget, 128));
            return text;
        }
        text.push_str(" retained explanation ");
        assert!(text.len() < 60 * 1024);
    }
}
fn compaction_request_options(request: &Value) -> String {
    let Value::Object(fields) = request else {
        panic!("request must be an object")
    };
    json::stringify(&Value::Object(
        fields
            .iter()
            .filter(|(key, _)| key != "messages")
            .cloned()
            .collect(),
    ))
    .unwrap()
}
#[test]
fn automatic_cli_compaction_observes_incoming_user_and_preserves_prior_summaries() {
    use vade_harness::{
        agent::{Conversation, Limits},
        context::Usage,
    };
    for disabled in [false, true] {
        let f = Fixture::new();
        std::fs::write(
            f.0.join("AGENTS.md"),
            "Keep exact scoped rules.\n  Spacing and 🦀 matter.\n",
        )
        .unwrap();
        let old_answer = format!(
            "old-result-marker {} old-result-tail",
            "resolved ".repeat(1000)
        );
        let (url, server) = mock(vec![
            answer().replace("mock complete", &old_answer),
            answer().replace("mock complete", &old_answer),
            answer(),
        ]);
        let output = run_bounded(
            compaction_cli_command(&f, &url).args(["--session", "automatic"]),
            "old constraints\nsecond constraints\nlatest constraints\n/quit\n",
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let initial_requests = server.join().unwrap();
        let workspace = vade_harness::tools::Workspace::new(&f.0).unwrap();
        let (journal, initial, _) =
            vade_harness::journal::Journal::resume(&workspace, "automatic", Limits::default())
                .unwrap();
        assert_eq!(journal.generation(), 0);
        drop(journal);
        let budget = Usage::for_conversation(&initial)
            .required(128)
            .saturating_mul(10)
            .div_ceil(7)
            .max(8192);
        assert!(!Usage::for_conversation(&initial).near_limit(budget, 128));
        let incoming =
            compaction_text_to_threshold(&initial, "user", "incoming-long-user-marker", 81, budget);
        let raw_before = std::fs::read(f.0.join(".vade/sessions/automatic.journal")).unwrap();
        let (url, server) = mock(vec![answer(), answer()]);
        let mut command = compaction_cli_command(&f, &url);
        command.args([
            "--resume",
            "automatic",
            "--context-tokens",
            &budget.to_string(),
        ]);
        if disabled {
            command.arg("--no-auto-compact");
        }
        let output = run_bounded(
            &mut command,
            &format!("{incoming}\nsmall followup\n/quit\n"),
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let requests = server.join().unwrap();
        let notice = String::from_utf8_lossy(&output.stderr);
        assert_eq!(notice.contains("Auto-compacted older results:"), !disabled);
        assert_eq!(
            notice.matches("Auto-compacted older results:").count(),
            usize::from(!disabled)
        );
        let prefix: Vec<_> = initial.messages()[..initial.prefix_len()]
            .iter()
            .map(|m| m.value().clone())
            .collect();
        let prefix_wire = json::stringify(&Value::Array(prefix)).unwrap();
        let initial_options = compaction_request_options(&initial_requests[0]);
        for request in &requests {
            let messages = request.get("messages").unwrap().as_array().unwrap();
            assert_eq!(
                json::stringify(&Value::Array(messages[..initial.prefix_len()].to_vec())).unwrap(),
                prefix_wire
            );
            assert_eq!(compaction_request_options(request), initial_options);
            assert!(
                messages
                    .iter()
                    .any(|m| m.get("content").and_then(Value::as_str) == Some(&incoming))
            );
            for constraints in [
                "old constraints",
                "second constraints",
                "latest constraints",
            ] {
                assert!(
                    messages
                        .iter()
                        .any(|m| m.get("content").and_then(Value::as_str) == Some(constraints))
                );
            }
        }
        let first_messages = requests[0].get("messages").unwrap().as_array().unwrap();
        let summaries: Vec<_> = first_messages
            .iter()
            .filter(|m| m.get("name").and_then(Value::as_str) == Some("vade_context_summary"))
            .cloned()
            .collect();
        assert_eq!(summaries.len(), usize::from(!disabled));
        assert_eq!(
            first_messages
                .iter()
                .filter(|m| m.get("content").and_then(Value::as_str) == Some(&old_answer))
                .count(),
            if disabled { 2 } else { 0 }
        );
        let raw_after = std::fs::read(f.0.join(".vade/sessions/automatic.journal")).unwrap();
        assert!(raw_after.starts_with(&raw_before));
        assert!(
            raw_after
                .windows(old_answer.len())
                .any(|bytes| bytes == old_answer.as_bytes())
        );
        let (journal, compacted, _) =
            vade_harness::journal::Journal::resume(&workspace, "automatic", Limits::default())
                .unwrap();
        assert_eq!(journal.generation(), u64::from(!disabled));
        drop(journal);
        if disabled {
            continue;
        }
        assert!(Usage::for_conversation(&compacted).required(128) <= budget * 3 / 5);
        assert!(
            requests[1]
                .get("messages")
                .unwrap()
                .as_array()
                .unwrap()
                .contains(&summaries[0])
        );

        // Grow through another threshold crossing, then keep sending small
        // turns above 60%. Earlier summary bytes must survive a new compaction
        // and must never become the next compaction's only removable input.
        let mut sizing = compacted.clone();
        sizing
            .push_user(compaction_message("user", "grow first"), Limits::default())
            .unwrap();
        let first_growth = compaction_text_to_threshold(
            &sizing,
            "assistant",
            "new-first-result-marker",
            70,
            budget,
        );
        let mut request =
            json::parse(&sizing.request(Limits::default().wire).to_json().unwrap()).unwrap();
        let Value::Object(fields) = &mut request else {
            panic!()
        };
        let Value::Array(messages) = &mut fields
            .iter_mut()
            .find(|(key, _)| key == "messages")
            .unwrap()
            .1
        else {
            panic!()
        };
        messages.push(compaction_message("assistant", &first_growth));
        sizing = Conversation::from_request(request, Limits::default()).unwrap();
        sizing
            .push_user(compaction_message("user", "grow second"), Limits::default())
            .unwrap();
        let second_growth = compaction_text_to_threshold(
            &sizing,
            "assistant",
            "new-second-result-marker",
            90,
            budget,
        );
        let (url, server) = mock(vec![
            answer().replace("mock complete", &first_growth),
            answer().replace("mock complete", &second_growth),
            answer(),
            answer(),
        ]);
        let output = run_bounded(
            compaction_cli_command(&f, &url).args([
                "--resume",
                "automatic",
                "--context-tokens",
                &budget.to_string(),
            ]),
            "grow first\ngrow second\ntrigger second compaction\nsmall tail\n/quit\n",
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let notice = String::from_utf8_lossy(&output.stderr);
        assert_eq!(notice.matches("Auto-compacted older results:").count(), 1);
        assert!(notice.contains("later admission overflow attempts another safe rewrite"));
        let requests = server.join().unwrap();
        for request in &requests {
            assert_eq!(compaction_request_options(request), initial_options);
            let messages = request.get("messages").unwrap().as_array().unwrap();
            assert!(messages.contains(&summaries[0]));
            assert_eq!(
                json::stringify(&Value::Array(messages[..initial.prefix_len()].to_vec())).unwrap(),
                prefix_wire
            );
        }
        let last = requests
            .last()
            .unwrap()
            .get("messages")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(
            last.iter()
                .filter(|m| m.get("name").and_then(Value::as_str) == Some("vade_context_summary"))
                .count(),
            2
        );
        assert!(
            !last
                .iter()
                .any(|m| m.get("content").and_then(Value::as_str) == Some(&first_growth))
        );
        assert!(
            last.iter()
                .any(|m| m.get("content").and_then(Value::as_str) == Some(&second_growth))
        );
        let raw_final = std::fs::read(f.0.join(".vade/sessions/automatic.journal")).unwrap();
        assert!(raw_final.starts_with(&raw_after));
        let (journal, resumed, _) =
            vade_harness::journal::Journal::resume(&workspace, "automatic", Limits::default())
                .unwrap();
        assert_eq!(journal.generation(), 2);
        assert!(
            resumed
                .messages()
                .iter()
                .any(|m| m.value() == &summaries[0])
        );
    }
}
