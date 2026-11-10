use super::*;
use crate::agent::{Conversation, Limits};
use crate::journal::Journal;
use crate::test_workspace::Fixture;
use std::time::Duration;

const SYSTEM: &str = r#"{"role":"system","content":"private system rules"},{"role":"developer","content":"private project context"}"#;
const TURN_ONE: &str = r#"{"role":"user","content":"Read the config"},{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"workspace_read","arguments":"{\"path\":\"a.conf\"}"}}]},{"role":"tool","tool_call_id":"call-1","content":"key = ```value```"},{"role":"assistant","content":"The key is set."}"#;
const TURN_TWO: &str = r#"{"role":"user","content":"Thanks"},{"role":"assistant","content":"Done."}"#;

fn conversation(messages: &[&str]) -> Conversation {
    let request = format!(r#"{{"model":"fixture-model","messages":[{}]}}"#, messages.join(","));
    Conversation::from_request(json::parse(&request).unwrap(), Limits::default()).unwrap()
}

fn touch(fixture: &Fixture, id: &str, seconds: u64) {
    let path = fixture.0.join(format!(".vade/sessions/{id}.journal"));
    let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    file.set_modified(UNIX_EPOCH + Duration::from_secs(seconds)).unwrap();
}

/// Two saved sessions: a named one from this build with two turns, and an
/// older unnamed one with one turn and no session record.
fn fixture() -> Fixture {
    let fixture = Fixture::new();
    let workspace = fixture.workspace();
    let mut named = Journal::create(&workspace, "alpha", &conversation(&[SYSTEM])).unwrap();
    named.record_details(Some("feature work"), Some(1_793_250_720_000)).unwrap();
    named.save(&conversation(&[SYSTEM, TURN_ONE])).unwrap();
    named.save(&conversation(&[SYSTEM, TURN_ONE, TURN_TWO])).unwrap();
    drop(named);
    let legacy = Journal::create(&workspace, "session-951868740000-7-0", &conversation(&[SYSTEM, TURN_TWO])).unwrap();
    drop(legacy);
    touch(&fixture, "alpha", 1_793_254_320);
    touch(&fixture, "session-951868740000-7-0", 951_868_800);
    fixture
}

#[test]
fn list_shows_both_sessions_newest_first() {
    let fixture = fixture();
    let listing = list(&fixture.workspace()).unwrap();
    assert!(listing.partial.is_empty());
    let list = listing.sessions.clone();
    let project = fixture.0.canonicalize().unwrap().to_string_lossy().into_owned();
    assert_eq!(list, vec![
        Summary { id: "alpha".into(), name: Some("feature work".into()), started_ms: Some(1_793_250_720_000), last_ms: Some(1_793_254_320_000), turns: 2, model: Some("fixture-model".into()), project: project.clone(), problem: None },
        Summary { id: "session-951868740000-7-0".into(), name: None, started_ms: Some(951_868_740_000), last_ms: Some(951_868_800_000), turns: 1, model: Some("fixture-model".into()), project, problem: None },
    ]);
    let text = table(&listing);
    let lines: Vec<_> = text.lines().collect();
    assert!(lines[0].starts_with("ID ") && lines[0].contains("LAST TURN") && lines[0].ends_with("PROJECT"));
    assert!(lines[1].starts_with("alpha ") && lines[1].contains("feature work") && lines[1].contains("2026-10-29 05:12 UTC") && lines[1].contains("2026-10-29 06:12 UTC"));
    assert!(lines[2].contains("2000-02-29 23:59 UTC") && lines[2].contains("2000-03-01 00:00 UTC"));
    // Listing a project with no saved sessions creates no state folder.
    let empty = Fixture::new();
    assert_eq!(super::list(&empty.workspace()).unwrap(), Listing::default());
    assert!(!empty.0.join(".vade").exists());
}

#[test]
fn resolve_takes_an_id_a_name_or_last_and_resume_replays_the_journal() {
    let fixture = fixture();
    let workspace = fixture.workspace();
    assert_eq!(resolve(&workspace, "session-951868740000-7-0", true).unwrap(), "session-951868740000-7-0");
    assert_eq!(resolve(&workspace, "feature work", true).unwrap(), "alpha");
    assert_eq!(resolve(&workspace, LAST, true).unwrap(), "alpha");
    assert!(resolve(&workspace, "missing", true).is_err());
    let (journal, restored, torn) = Journal::resume(&workspace, &resolve(&workspace, LAST, true).unwrap(), Limits::default()).unwrap();
    assert!(!torn);
    assert_eq!(restored.messages(), conversation(&[SYSTEM, TURN_ONE, TURN_TWO]).messages());
    // An open session is not "last" for a resume, but an export may still read it.
    assert_eq!(resolve(&workspace, LAST, true).unwrap(), "session-951868740000-7-0");
    assert_eq!(resolve(&workspace, LAST, false).unwrap(), "alpha");
    drop(journal);
    // A shared name selects the newest session; a rename on resume wins.
    let mut other = Journal::create(&workspace, "beta", &conversation(&[SYSTEM, TURN_TWO])).unwrap();
    other.record_details(Some("feature work"), Some(1)).unwrap();
    drop(other);
    touch(&fixture, "beta", 1_793_260_000);
    assert_eq!(resolve(&workspace, "feature work", true).unwrap(), "beta");
    let (mut renamed, _, _) = Journal::resume(&workspace, "alpha", Limits::default()).unwrap();
    renamed.record_details(Some("renamed"), None).unwrap();
    drop(renamed);
    let alpha = super::list(&workspace).unwrap().sessions.into_iter().find(|s| s.id == "alpha").unwrap();
    assert_eq!((alpha.name.as_deref(), alpha.started_ms), (Some("renamed"), Some(1_793_250_720_000)));
}

#[test]
fn export_writes_a_readable_markdown_transcript() {
    let fixture = fixture();
    let workspace = fixture.workspace();
    let text = export_markdown(&workspace, "alpha").unwrap();
    assert!(text.starts_with("# Session feature work\n\n- ID: `alpha`\n- Name: feature work\n- Model: fixture-model\n"), "{text}");
    assert!(text.contains("- Started: 2026-10-29 05:12 UTC\n- Last turn: 2026-10-29 06:12 UTC\n- Turns: 2\n"));
    assert!(!text.contains("private system rules") && !text.contains("private project context"));
    let body = text.split_once("- Turns: 2\n\n").unwrap().1;
    assert_eq!(body, "## User\n\nRead the config\n\n## Assistant\n\nTool call `workspace_read`:\n\n```json\n{\"path\":\"a.conf\"}\n```\n\nTool result:\n\n````text\nkey = ```value```\n````\n\n## Assistant\n\nThe key is set.\n\n## User\n\nThanks\n\n## Assistant\n\nDone.\n\n");
    // The CLI resolves names and last the same way.
    let root = fixture.0.to_string_lossy().into_owned();
    let args = |extra: &[&str]| extra.iter().map(|s| s.to_string()).chain(["--workspace".into(), root.clone(), "--state-in-project".into()]).collect::<Vec<_>>();
    assert_eq!(run(args(&["export", "feature work", "--markdown"])).unwrap(), text);
    assert_eq!(run(args(&["export", "last"])).unwrap(), text);
    assert!(run(args(&[])).unwrap().lines().nth(1).unwrap().starts_with("alpha "));
    assert_eq!(run(args(&["export"])).unwrap_err().0, 2);
    assert_eq!(run(args(&["delete", "a", "b"])).unwrap_err().0, 2);
    assert_eq!(run(args(&["--all"])).unwrap_err().0, 2);
    assert_eq!(run(args(&["export", "missing"])).unwrap_err().0, 1);
}

#[test]
fn delete_refuses_an_open_session_and_removes_a_closed_one() {
    let fixture = fixture();
    let workspace = fixture.workspace();
    let (open, _, _) = Journal::resume(&workspace, "alpha", Limits::default()).unwrap();
    assert!(delete(&workspace, "alpha").unwrap_err().to_string().contains("open in another chat"));
    drop(open);
    delete(&workspace, "alpha").unwrap();
    assert!(delete(&workspace, "alpha").is_err());
    assert!(delete(&workspace, "../escape").is_err());
    let ids: Vec<_> = super::list(&workspace).unwrap().sessions.into_iter().map(|s| s.id).collect();
    assert_eq!(ids, ["session-951868740000-7-0"]);
    assert!(Journal::resume(&workspace, "alpha", Limits::default()).is_err());
}

#[test]
fn compaction_keeps_old_turns_in_the_export_and_resume_still_works() {
    let fixture = Fixture::new();
    let workspace = fixture.workspace();
    let long = format!(r#"{{"role":"user","content":"First question"}},{{"role":"assistant","content":"{}"}}"#, "old detail ".repeat(200));
    let before = conversation(&[SYSTEM, &long, TURN_TWO]);
    let mut journal = Journal::create(&workspace, "compacted", &before).unwrap();
    let plan = crate::context::plan(&before, "The user asked a first question.", 1, Limits::default()).unwrap();
    journal.compact(&plan).unwrap();
    let mut after = plan.conversation().clone();
    after.push_user(json::parse(r#"{"role":"user","content":"Next question"}"#).unwrap(), Limits::default()).unwrap();
    journal.save(&after).unwrap();
    drop(journal);
    let (_, restored, _) = Journal::resume(&workspace, "compacted", Limits::default()).unwrap();
    assert_eq!(restored.messages(), after.messages());
    let summary = super::list(&workspace).unwrap().sessions.remove(0);
    assert_eq!(summary.turns, 3);
    let text = export_markdown(&workspace, "compacted").unwrap();
    let first = text.find("First question").unwrap();
    let note = text.find("*The context was compacted here.").unwrap();
    let next = text.find("Next question").unwrap();
    assert!(first < note && note < next && text.contains("old detail old detail"), "{text}");
    assert!(text.contains("```text\nThe user asked a first question.\n```"), "{text}");
    assert_eq!(text.matches("## User\n\nThanks").count(), 1, "{text}");
}

fn frames(records: &[&str], torn: &[u8]) -> Vec<u8> {
    let mut raw = Vec::new();
    for record in records {
        let len = (record.len() as u64).to_le_bytes();
        raw.extend_from_slice(&len);
        raw.extend_from_slice(record.as_bytes());
        raw.extend_from_slice(&len);
    }
    raw.extend_from_slice(torn);
    raw
}

#[test]
fn transcript_tolerates_unknown_records_rewrites_and_a_torn_tail() {
    let note = json::stringify(&Value::String(RESUME_NOTE.into())).unwrap();
    let raw = frames(&[
        r#"{"v":1,"seq":0,"keep":0,"messages":[{"role":"system","content":"s"}],"options":{"model":"m"},"workspace":"w"}"#,
        r#"{"v":1,"seq":1,"keep":1,"messages":[],"session":{"name":"n","started_ms":5}}"#,
        // A staged step and its later revision: only the revision is shown.
        r#"{"v":1,"seq":2,"keep":1,"messages":[{"role":"user","content":"q1"},{"role":"assistant","content":"draft"}]}"#,
        r#"{"v":1,"seq":3,"keep":2,"messages":[{"role":"assistant","content":"a1"}]}"#,
        r#"{"v":1,"seq":4,"keep":3,"messages":[],"future_kind":{"anything":[1,2,3]}}"#,
        // A context rewrite in a shape this build does not know.
        r#"{"v":1,"seq":5,"keep":0,"messages":[{"role":"system","content":"s"},{"role":"user","content":"summary of q1"}],"rollover":{"kept_turns":0}}"#,
        &format!(r#"{{"v":1,"seq":6,"keep":2,"messages":[{{"role":"user","content":{note}}},{{"role":"user","content":"q2"}},{{"role":"assistant","content":"a2"}}]}}"#),
    ], &[9, 0, 0, 0, 0, 0, 0, 0, b'{']);
    let records = records(&raw).unwrap();
    assert_eq!(records.len(), 7);
    let t = transcript(&records).unwrap();
    assert_eq!((t.name.as_deref(), t.started_ms, t.model.as_deref(), t.turns()), (Some("n"), Some(5), Some("m"), 2));
    let roles: Vec<_> = t.entries.iter().map(|e| match e {
        Entry::Message(m) => format!("{}:{}", m.get("role").and_then(Value::as_str).unwrap(), m.get("content").and_then(Value::as_str).unwrap()),
        Entry::Compacted(s) => format!("compacted:{s:?}"),
        Entry::Resumed => "resumed".into(),
        Entry::Note(name, _) => format!("note:{name}"),
    }).collect();
    assert_eq!(roles, ["system:s", "user:q1", "assistant:a1", "compacted:None", "resumed", "user:q2", "assistant:a2"]);
    // A damaged complete record is an error, not a silent gap.
    let mut bad = frames(&[r#"{"v":1,"seq":0,"keep":0,"messages":[]}"#], &[]);
    let last = bad.len() - 1;
    bad[last] ^= 1;
    assert!(super::records(&bad).is_err());
    assert!(transcript(&super::records(&frames(&[r#"{"v":1,"seq":0,"keep":3,"messages":[]}"#], &[])).unwrap()).is_err());
}

#[test]
fn list_home_covers_every_project_and_marks_unreadable_journals() {
    let home = Fixture::new();
    let project = Fixture::new();
    let state = StateLocation::home(home.0.clone()).unwrap();
    let workspace = Workspace::open(&project.0, state).unwrap();
    drop(Journal::create(&workspace, "home-one", &conversation(&[SYSTEM, TURN_TWO])).unwrap());
    let dir = workspace.session_dir().unwrap();
    dir.write_atomic("broken.journal", b"\x05\0\0\0\0\0\0\0{bad}\x05\0\0\0\0\0\0\0").unwrap();
    let listing = list_home(&home.0).unwrap();
    let list = &listing.sessions;
    assert_eq!(list.len(), 2);
    let project = project.0.canonicalize().unwrap().to_string_lossy().into_owned();
    assert!(list.iter().all(|s| s.project == project));
    let broken = list.iter().find(|s| s.id == "broken").unwrap();
    assert!(broken.problem.is_some());
    assert!(table(&listing).contains("cannot resume: journal record corrupt"));
    let root = project.clone();
    let home_arg = home.0.to_string_lossy().into_owned();
    assert!(run(vec!["--all".into(), "--vade-home".into(), home_arg.clone()]).unwrap().contains("home-one"));
    assert!(run(vec!["list".into(), "--workspace".into(), root, "--vade-home".into(), home_arg]).unwrap().contains("home-one"));
}

#[test]
fn names_and_times() {
    for good in ["feature work", "bug-42", "été"] {
        assert!(valid_display_name(good), "{good}");
    }
    // A name that starts with a hyphen reads as an option on the command line.
    for bad in ["", " lead", "trail ", "last", "-h", "--all", "-x", "a/b", "a\\b", "tab\there", &"x".repeat(65), "\u{202e}rtl"] {
        assert!(!valid_display_name(bad), "{bad:?}");
    }
    assert_eq!(time(Some(0)), "1970-01-01 00:00 UTC");
    assert_eq!(time(None), "-");
}

#[test]
fn a_partial_scan_is_reported_and_never_picks_a_session_by_name_or_last() {
    let fixture = fixture();
    let workspace = fixture.workspace();
    // A delete keeps the empty lock file, and lock files count toward the scan limit.
    for n in 0..MAX_ENTRIES {
        std::fs::File::create(fixture.0.join(format!(".vade/sessions/gone-{n}.journal-lock"))).unwrap();
    }
    let listing = super::list(&workspace).unwrap();
    assert_eq!(listing.partial.len(), 1);
    let text = table(&listing);
    assert!(text.contains("more than 4096 entries") && text.contains("may miss sessions"), "{text}");
    for query in [LAST, "feature work"] {
        for resume in [true, false] {
            let error = resolve(&workspace, query, resume).unwrap_err().to_string();
            assert!(error.contains("more than 4096 entries") && error.contains("by ID"), "{error}");
        }
    }
    // An exact ID needs no scan.
    assert_eq!(resolve(&workspace, "alpha", true).unwrap(), "alpha");
    let root = fixture.0.to_string_lossy().into_owned();
    assert!(run(vec!["--workspace".into(), root.clone(), "--state-in-project".into()]).unwrap().contains("may miss sessions"));
    assert_eq!(run(vec!["export".into(), "last".into(), "--workspace".into(), root, "--state-in-project".into()]).unwrap_err().0, 1);
    // The project scan of a Vade home reports its limit too.
    let home = Fixture::new();
    for n in 0..=MAX_PROJECTS {
        std::fs::create_dir_all(home.0.join(format!("projects/p-{n}"))).unwrap();
    }
    let listing = list_home(&home.0).unwrap();
    assert!(listing.sessions.is_empty() && listing.partial.len() == 1);
    assert!(table(&listing).contains("more than 1024 projects"), "{}", table(&listing));
}

const OPTIONS: &str = r#"{"v":1,"seq":0,"keep":0,"messages":[{"role":"system","content":"s"}],"options":{"model":"m"}}"#;

#[test]
fn transcript_applies_the_replay_record_checks() {
    let read = |records: &[&str]| transcript(&super::records(&frames(records, &[])).unwrap()).map(|t| t.turns());
    let turn = r#"[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]"#;
    assert_eq!(read(&[OPTIONS, &format!(r#"{{"v":1,"seq":1,"keep":1,"messages":{turn},"new_key":{{"x":1}}}}"#)]).unwrap(), 1);
    for bad in [
        format!(r#"{{"v":1,"seq":2,"keep":1,"messages":{turn}}}"#),
        format!(r#"{{"v":2,"seq":1,"keep":1,"messages":{turn}}}"#),
        format!(r#"{{"seq":1,"keep":1,"messages":{turn}}}"#),
        r#"{"v":1,"seq":1,"keep":1}"#.into(),
        r#"{"v":1,"seq":1,"keep":1,"messages":{"role":"user","content":"q"}}"#.into(),
        format!(r#"{{"v":1,"seq":1,"keep":1,"messages":{turn},"options":{{"model":"other"}}}}"#),
        // A compaction record must name the record and history it replaces.
        r#"{"v":1,"seq":1,"keep":0,"messages":[],"compaction":{"generation":2,"source_record":0,"source_messages":1}}"#.into(),
        r#"{"v":1,"seq":1,"keep":0,"messages":[],"compaction":{"generation":1,"source_record":0,"source_messages":7}}"#.into(),
    ] {
        assert!(read(&[OPTIONS, &bad]).is_err(), "{bad}");
    }
    assert!(read(&[r#"{"v":1,"seq":0,"keep":0,"messages":[]}"#]).is_err());
    // A model-written rollover adds metadata keys; they are accepted.
    let rollover = r#"{"v":1,"seq":2,"keep":0,"messages":[{"role":"system","content":"s"},{"role":"user","name":"vade_context_summary","content":"sum"}],"compaction":{"generation":1,"source_record":1,"source_messages":3,"summary":"sum","summary_origin":"model","kept_turns":1,"summary_tokens":12}}"#;
    let t = transcript(&super::records(&frames(&[OPTIONS, &format!(r#"{{"v":1,"seq":1,"keep":1,"messages":{turn}}}"#), rollover], &[])).unwrap()).unwrap();
    assert_eq!(t.entries.last(), Some(&Entry::Compacted(Some("sum".into()))));
}

#[test]
fn a_journal_that_resume_rejects_is_unreadable_and_last_skips_it() {
    let fixture = fixture();
    let workspace = fixture.workspace();
    let dir = workspace.session_dir().unwrap();
    // Readable framing, a user turn, and a sequence number that resume rejects.
    dir.write_atomic("broken.journal", &frames(&[OPTIONS, r#"{"v":1,"seq":4,"keep":1,"messages":[{"role":"user","content":"q"}]}"#], &[])).unwrap();
    touch(&fixture, "broken", 1_893_456_000);
    let listing = super::list(&workspace).unwrap();
    let broken = listing.sessions.iter().find(|s| s.id == "broken").unwrap();
    assert_eq!(broken.problem.as_deref(), Some("journal version/sequence mismatch"));
    assert!(Journal::resume(&workspace, "broken", Limits::default()).is_err());
    assert_eq!(resolve(&workspace, LAST, true).unwrap(), "alpha");
    assert!(export_markdown(&workspace, "broken").is_err());
}

#[test]
fn export_shows_refusals_and_host_notes() {
    let raw = frames(&[
        OPTIONS,
        r#"{"v":1,"seq":1,"keep":1,"messages":[{"role":"user","content":"Do the thing"},{"role":"assistant","content":"","refusal":"I cannot do that thing."},{"role":"user","name":"vade_project_instructions","content":"Project instructions changed between turns."},{"role":"user","content":"Then this"},{"role":"assistant","content":null,"refusal":[{"type":"refusal","text":"Not this either."}]}]}"#,
    ], &[]);
    let t = transcript(&super::records(&raw).unwrap()).unwrap();
    // A note the host adds for the model is not a user turn.
    assert_eq!(t.turns(), 2);
    let text = markdown("r", "/p", &t, None, None);
    let body = text.split_once("- Turns: 2\n\n").unwrap().1;
    assert_eq!(body, "## User\n\nDo the thing\n\n## Assistant\n\nI cannot do that thing.\n\n*Vade added a note for the model here (vade_project_instructions):*\n\n```text\nProject instructions changed between turns.\n```\n\n## User\n\nThen this\n\n## Assistant\n\nNot this either.\n\n");
}

#[test]
fn the_list_accepts_exactly_what_resume_accepts() {
    let fixture = fixture();
    let workspace = fixture.workspace();
    let dir = workspace.session_dir().unwrap();
    let turn = r#"{"v":1,"seq":1,"keep":1,"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}"#;
    let options_null = r#"{"v":1,"seq":0,"keep":0,"messages":[{"role":"system","content":"s"}],"options":null}"#;
    let tool_without_call = r#"{"v":1,"seq":1,"keep":1,"messages":[{"role":"user","content":"q"},{"role":"tool","content":"x"}]}"#;
    let mut many: Vec<String> = vec![OPTIONS.into(), turn.into()];
    many.extend((2..=8192).map(|seq| format!(r#"{{"v":1,"seq":{seq},"keep":3,"messages":[]}}"#)));
    let many: Vec<&str> = many.iter().map(String::as_str).collect();
    let none = "session has no complete initial record";
    let cases: Vec<(&str, Vec<u8>, &str)> = vec![
        ("empty", Vec::new(), none),
        ("torn-first", frames(&[], &frames(&[OPTIONS], &[])[..20]), none),
        ("options-null", frames(&[options_null, turn], &[]), none),
        ("bad-message", frames(&[OPTIONS, tool_without_call], &[]), "missing or non-string required field"),
        ("too-many", frames(&many, &[]), "session record count exceeds limit"),
        // Record 0 names another folder: a fresh clone at the same path.
        ("other-folder", frames(&[r#"{"v":1,"seq":0,"keep":0,"messages":[{"role":"system","content":"s"}],"options":{"model":"m"},"workspace":"1:2"}"#, turn], &[]), "journal belongs to a different workspace directory"),
    ];
    for (n, (id, raw, _)) in cases.iter().enumerate() {
        dir.write_atomic(&format!("{id}.journal"), raw).unwrap();
        touch(&fixture, id, 1_893_456_000 + n as u64);
    }
    // A torn final record after a complete one is what resume truncates; it stays usable.
    dir.write_atomic("torn-tail.journal", &frames(&[OPTIONS, turn], &frames(&[r#"{"v":1,"seq":2,"keep":3,"messages":[]}"#], &[])[..9])).unwrap();
    touch(&fixture, "torn-tail", 1_793_250_000);
    let listing = super::list(&workspace).unwrap();
    for (id, _, problem) in &cases {
        let row = listing.sessions.iter().find(|s| s.id == *id).unwrap();
        assert_eq!(row.problem.as_deref(), Some(*problem), "{id}");
        assert!(Journal::resume(&workspace, id, Limits::default()).is_err(), "{id} resumes");
    }
    assert!(table(&listing).contains("cannot resume: "));
    assert_eq!(resolve(&workspace, LAST, true).unwrap(), "alpha");
    // A journal from another folder still exports; a damaged one does not.
    assert!(export_markdown(&workspace, "other-folder").unwrap().contains("## User\n\nq\n\n"));
    assert!(export_markdown(&workspace, "options-null").is_err());
    // Only a resume truncates a torn tail; the list leaves the file as it is.
    let torn_len = std::fs::metadata(fixture.0.join(".vade/sessions/torn-tail.journal")).unwrap().len();
    assert_eq!(torn_len as usize, frames(&[OPTIONS, turn], &[]).len() + 9);
    let torn = listing.sessions.iter().find(|s| s.id == "torn-tail").unwrap();
    assert_eq!((torn.problem.as_deref(), torn.turns), (None, 1));
    assert!(Journal::resume(&workspace, "torn-tail", Limits::default()).is_ok());
}

#[test]
fn a_folder_at_the_scan_limit_is_complete() {
    let fixture = fixture();
    let sessions = fixture.0.join(".vade/sessions");
    let present = std::fs::read_dir(&sessions).unwrap().count();
    for n in present..MAX_ENTRIES {
        std::fs::File::create(sessions.join(format!("gone-{n}.journal-lock"))).unwrap();
    }
    assert_eq!(std::fs::read_dir(&sessions).unwrap().count(), MAX_ENTRIES);
    let listing = super::list(&fixture.workspace()).unwrap();
    assert!(listing.partial.is_empty(), "{:?}", listing.partial);
    assert_eq!(resolve(&fixture.workspace(), LAST, true).unwrap(), "alpha");
    let home = Fixture::new();
    for n in 0..MAX_PROJECTS {
        std::fs::create_dir_all(home.0.join(format!("projects/p-{n}"))).unwrap();
    }
    let partial = list_home(&home.0).unwrap().partial;
    assert!(partial.is_empty(), "{:?}", &partial[..partial.len().min(2)]);
}

#[test]
fn last_says_why_no_session_qualifies() {
    let fixture = Fixture::new();
    let workspace = fixture.workspace();
    assert!(resolve(&workspace, LAST, true).unwrap_err().to_string().contains("has no saved sessions"));
    drop(Journal::create(&workspace, "empty", &conversation(&[SYSTEM])).unwrap());
    let error = resolve(&workspace, LAST, true).unwrap_err().to_string();
    assert!(error.contains("no saved session") && error.contains("can be resumed") && error.contains("vade sessions"), "{error}");
}

#[test]
fn list_home_reports_a_session_folder_it_cannot_open() {
    use std::os::unix::fs::PermissionsExt;
    let home = Fixture::new();
    let project = Fixture::new();
    let workspace = Workspace::open(&project.0, StateLocation::home(home.0.clone()).unwrap()).unwrap();
    drop(Journal::create(&workspace, "kept", &conversation(&[SYSTEM, TURN_TWO])).unwrap());
    let locked = home.0.join("projects/locked-1/sessions");
    std::fs::create_dir_all(&locked).unwrap();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    let listing = list_home(&home.0).unwrap();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(listing.sessions.iter().any(|s| s.id == "kept"));
    assert_eq!(listing.partial.len(), 1, "{:?}", listing.partial);
    assert!(listing.partial[0].contains("Cannot read"), "{:?}", listing.partial);
}
