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
    assert_eq!(fork.checkpoint_hints(), std::slice::from_ref(&hint));
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
