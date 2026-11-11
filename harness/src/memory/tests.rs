use super::*;
use crate::test_workspace::Fixture;

#[test]
fn upsert_replace_forget_and_file_round_trip() {
    let fixture = Fixture::new();
    let workspace = fixture.workspace();
    assert_eq!(Memory::load(&workspace).unwrap().content(), "");
    assert_eq!(change(&workspace, "response-style", Some("Use short answers.")).unwrap(), 1);
    assert_eq!(change(&workspace, "decision", Some("Keep the release small.")).unwrap(), 2);
    assert_eq!(change(&workspace, "response-style", Some("Explain tradeoffs.")).unwrap(), 2);
    let memory = Memory::load(&workspace).unwrap();
    assert!(!memory.content().contains("Use short answers."));
    let content = memory.content();
    let line = content.lines().last().unwrap();
    let date = line.strip_prefix("- response-style: Explain tradeoffs. (").unwrap().strip_suffix(')').unwrap();
    assert!(valid_date(date));
    assert_eq!(fixture.read(".vade/MEMORY.md"), memory.content());
    assert_eq!(change(&workspace, "decision", None).unwrap(), 1);
    assert_eq!(change(&workspace, "missing", None).unwrap(), 1);
    assert_eq!(change(&workspace, "response-style", None).unwrap(), 0);
    assert_eq!(Memory::load(&workspace).unwrap().content(), "");
}

#[test]
fn newest_entries_win_at_both_caps() {
    let fixture = Fixture::new();
    let workspace = fixture.workspace();
    for n in 0..70 { change(&workspace, &format!("note-{n}"), Some("Keep this.")).unwrap(); }
    let memory = Memory::load(&workspace).unwrap();
    assert_eq!(memory.count(), MAX_ENTRIES);
    assert!(memory.content().starts_with(MARKER));
    assert!(!memory.content().contains("- note-5:"));
    assert!(memory.content().contains("- note-6:"));
    for n in 0..6 { change(&workspace, &format!("large-{n}"), Some(&"é".repeat(1000))).unwrap(); }
    let memory = Memory::load(&workspace).unwrap();
    assert!(memory.content().len() <= MAX_BYTES);
    assert!(memory.content().contains("- large-5:"));
    assert!(!memory.content().contains("- large-0:"));
    assert!(fixture.read(".vade/MEMORY.md").len() <= MAX_BYTES);
    assert!(change(&workspace, "too-large", Some(&"x".repeat(MAX_BYTES))).is_err());
}

#[test]
fn hand_edited_files_keep_newest_complete_lines_and_skip_bad_lines() {
    let fixture = Fixture::new();
    let workspace = fixture.workspace();
    let dir = workspace.state_dir().unwrap();
    let mut text = String::from("malformed\n- Bad_key: no (2026-01-01)\n- bad-date: no (2026-02-30)\n");
    for n in 0..100 { text.push_str(&format!("- note-{n}: {} (2026-01-01)\n", "é".repeat(90))); }
    text.push_str("- final: Keep this. (2026-10-29)\n");
    dir.write_atomic("MEMORY.md", text.as_bytes()).unwrap();
    let memory = Memory::load(&workspace).unwrap();
    assert!(memory.count() <= MAX_ENTRIES);
    assert!(memory.content().len() <= MAX_BYTES);
    assert!(memory.content().starts_with(MARKER));
    assert!(memory.content().ends_with("- final: Keep this. (2026-10-29)\n"));
    assert!(!memory.content().contains("note-0:"));
    dir.write_atomic("MEMORY.md", b"bad\n- good: Keep this. (2026-10-29)\n- no-date: skipped\n\xff\n").unwrap();
    assert_eq!(Memory::load(&workspace).unwrap().count(), 1);
    dir.write_atomic("MEMORY.md", b"").unwrap();
    assert_eq!(Memory::load(&workspace).unwrap().content(), "");
    let rows: String = (0..65).map(|n| format!("- n-{n}: x (2026-01-01)\n")).collect();
    dir.write_atomic("MEMORY.md", rows.as_bytes()).unwrap();
    let memory = Memory::load(&workspace).unwrap();
    assert_eq!(memory.count(), 64);
    assert!(!memory.content().contains("- n-0:"));
}

#[test]
fn prompt_label_empty_notes_switches_and_dates() {
    assert_eq!(with_memory("system", ""), "system");
    let prompt = with_memory("system", "- style: Brief. (2026-10-29)\n");
    assert!(prompt.contains(LABEL));
    assert_eq!(without_memory(&prompt), "system");
    assert!(enabled(false, None));
    assert!(enabled(false, Some("1")));
    assert!(!enabled(false, Some("0")));
    assert!(!enabled(true, Some("1")));
    assert_eq!(date_at(0).unwrap(), "1970-01-01");
    assert_eq!(date_at(11016).unwrap(), "2000-02-29");
    assert_eq!(date_at(20755).unwrap(), "2026-10-29");
    assert!(!valid_date("1900-02-29"));
    assert!(valid_date("2000-02-29"));
    for key in ["", "Bad", "a--b", "-key", "key-", "a_b"] { assert!(!valid_key(key)); }
    assert!(!valid_text("first\nsecond"));
}

#[test]
fn tools_are_opt_in_and_return_counts() {
    use crate::{agent::{ToolExecutor, ToolInvocation}, agent_wire::ToolCall, coding_tools::CodingTools, shell::ApprovalPolicy};
    use std::sync::atomic::AtomicBool;
    let fixture = Fixture::new();
    let workspace = fixture.workspace();
    let cancel = AtomicBool::new(false);
    let mut tools = CodingTools::new(&workspace, ApprovalPolicy::Auto, |_, _| true);
    let invoke = |tools: &mut CodingTools<'_, _>, name: &str, args: &str| {
        let call = ToolCall { id: "memory-call".into(), name: name.into(), arguments: args.into() };
        let arguments = json::parse(args).unwrap();
        tools.execute(ToolInvocation { call: &call, arguments: &arguments, max_output_bytes: 4096 }, &cancel)
    };
    assert!(invoke(&mut tools, "memory_save", r#"{"key":"style","text":"Brief."}"#).is_err());
    let mut tools = tools.with_memory(true);
    assert_eq!(invoke(&mut tools, "memory_save", r#"{"key":"style","text":"Brief."}"#).unwrap(), r#"{"entries":1}"#);
    assert_eq!(invoke(&mut tools, "memory_forget", r#"{"key":"style"}"#).unwrap(), r#"{"entries":0}"#);
    assert!(invoke(&mut tools, "memory_save", r#"{"key":"style","text":"Brief.","extra":true}"#).is_err());
    let mut definitions = crate::coding_tools::definitions();
    assert!(!json::stringify(&definitions).unwrap().contains("memory_save"));
    add_tools(&mut definitions);
    assert!(json::stringify(&definitions).unwrap().contains("memory_save"));
}

#[test]
fn quality_card_requests_are_identical_with_project_memory() {
    let fixture = Fixture::new();
    let workspace = fixture.workspace();
    let config = crate::quality::Config::default();
    let before: Vec<_> = crate::quality::CASES.iter().map(|case| crate::quality::request_body(&config, case, None).unwrap()).collect();
    change(&workspace, "card-sentinel", Some("This note must never reach a card.")).unwrap();
    for (case, before) in crate::quality::CASES.iter().zip(before) {
        let after = crate::quality::request_body(&config, case, None).unwrap();
        assert_eq!(before, after);
        assert!(!after.contains("card-sentinel"));
        assert!(!after.contains(LABEL));
    }
}
