//! Byte-level edit regressions. All writes stay in a private fixture.
use std::{fs, path::PathBuf, sync::atomic::{AtomicBool, AtomicUsize, Ordering}};
use vade_harness::{agent::{ToolExecutor, ToolInvocation}, agent_wire::ToolCall, coding_tools::CodingTools, edit, json, shell::ApprovalPolicy, tools::Workspace};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!("vade-edit-regressions-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir(&p).unwrap(); Self(p)
    }
    fn write(&self, name: &str, bytes: impl AsRef<[u8]>) { fs::write(self.0.join(name), bytes).unwrap(); }
    fn read(&self, name: &str) -> Vec<u8> { fs::read(self.0.join(name)).unwrap() }
    fn workspace(&self) -> Workspace { Workspace::new(&self.0).unwrap() }
}
impl Drop for Fixture { fn drop(&mut self) { fs::remove_dir_all(&self.0).unwrap(); } }
fn patch(body: &str) -> String { format!("*** Begin Patch\n{body}\n*** End Patch") }
fn call(tools: &mut impl ToolExecutor, name: &str, arguments: json::Value) -> String {
    let call = ToolCall { id: "regression".into(), name: name.into(), arguments: json::stringify(&arguments).unwrap() };
    tools.execute(ToolInvocation { call: &call, arguments: &arguments, max_output_bytes: 4096 }, &AtomicBool::new(false)).unwrap()
}
#[test]
fn overlapping_old_blocks_are_refused_without_writes() {
    let f = Fixture::new(); f.write("a", "foo\nfoo\nfoo\n");
    let result = edit::prepare(&f.workspace(), &patch("*** Update File: a\n@@\n-foo\n-foo\n+bar"));
    assert!(result.is_err(), "overlapping old blocks were accepted");
    assert_eq!(f.read("a"), b"foo\nfoo\nfoo\n");
}
/// Whether this temp volume folds case, found by creating a file in a private
/// temp folder and opening it with its case flipped. Not by target OS: a Mac
/// volume can be case-sensitive and a Linux folder can fold case.
fn temp_disk_folds_case() -> bool {
    let probe = Fixture::new(); probe.write("CaseProbe", "x");
    match fs::symlink_metadata(probe.0.join("cASEpROBE")) {
        Ok(_) => true,
        Err(e) => { assert_eq!(e.kind(), std::io::ErrorKind::NotFound); false }
    }
}
fn apply(f: &Fixture, body: &str) -> Result<String, String> {
    edit::prepare(&f.workspace(), &patch(body)).map_err(|e| e.to_string())?
        .apply(&AtomicBool::new(false)).map_err(|e| e.to_string())
}
#[test]
fn case_variant_parents_follow_the_disk() {
    let f = Fixture::new(); f.write("keep", "x");
    let result = apply(&f, "*** Add File: Dir/a\n+one\n*** Add File: dir/b\n+two");
    if temp_disk_folds_case() {
        assert!(result.unwrap_err().contains("this disk ignores case"));
        assert!(!f.0.join("Dir").exists());
    } else {
        result.unwrap();
        assert_eq!(f.read("Dir/a"), b"one\n");
        assert_eq!(f.read("dir/b"), b"two\n");
    }
}
#[test]
fn case_variant_files_follow_the_disk() {
    // An empty workspace: nothing inside it to probe.
    let f = Fixture::new();
    let result = apply(&f, "*** Add File: First\n+one\n*** Add File: first\n+two");
    if temp_disk_folds_case() {
        assert!(result.unwrap_err().contains("this disk ignores case"));
        assert_eq!(fs::read_dir(&f.0).unwrap().count(), 0);
    } else {
        result.unwrap();
        assert_eq!(f.read("First"), b"one\n");
        assert_eq!(f.read("first"), b"two\n");
    }
}
#[test]
fn case_variant_of_existing_files_follows_the_disk() {
    let f = Fixture::new(); f.write("Name", "old\n");
    let result = apply(&f, "*** Update File: Name\n@@\n-old\n+new\n*** Add File: name\n+added");
    if temp_disk_folds_case() {
        assert!(result.is_err());
        assert_eq!(f.read("Name"), b"old\n");
        assert_eq!(fs::read_dir(&f.0).unwrap().count(), 1);
    } else {
        result.unwrap();
        assert_eq!(f.read("Name"), b"new\n");
        assert_eq!(f.read("name"), b"added\n");
        // Two existing entries that differ only by case are two files here.
        apply(&f, "*** Update File: Name\n@@\n-new\n+newer\n*** Update File: name\n@@\n-added\n+again").unwrap();
        assert_eq!(f.read("Name"), b"newer\n");
        assert_eq!(f.read("name"), b"again\n");
    }
}

#[test]
fn move_destination_cannot_be_another_changes_parent() {
    let f = Fixture::new(); f.write("source", "original\n");
    let result = edit::prepare(&f.workspace(), &patch("*** Add File: target/child\n+one\n*** Update File: source\n*** Move to: target"));
    assert!(result.is_err(), "move destination overlaps a new parent");
    assert_eq!(f.read("source"), b"original\n");
    assert!(!f.0.join("target").exists());
}

#[test]
fn move_destination_case_variant_follows_the_disk() {
    let f = Fixture::new(); f.write("source", "original\n");
    let result = apply(&f, "*** Add File: Target\n+one\n*** Update File: source\n*** Move to: target");
    if temp_disk_folds_case() {
        assert!(result.is_err(), "move destination aliases an added file");
        assert_eq!(f.read("source"), b"original\n");
        assert!(!f.0.join("Target").exists());
    } else {
        result.unwrap();
        assert_eq!(f.read("Target"), b"one\n");
        assert_eq!(f.read("target"), b"original\n");
        assert!(!f.0.join("source").exists());
    }
}

#[test]
fn unique_line_start_match_inside_a_substring_match_is_used() {
    // Substring search finds "a\na\n" first at byte 1 (not a line start) and
    // skips the one valid place, byte 3. The line-start scan finds byte 3.
    let f = Fixture::new(); f.write("a", "xa\na\na\n");
    apply(&f, "*** Update File: a\n@@\n-a\n-a\n+b").unwrap();
    assert_eq!(f.read("a"), b"xa\nb\n");
}

#[test]
fn loose_overlapping_blocks_are_also_ambiguous() {
    let f = Fixture::new(); f.write("a", "x  y\nx  y\nx  y\n");
    let result = edit::prepare(&f.workspace(), &patch("*** Update File: a\n@@\n-x y\n-x y\n+new"));
    assert!(result.err().unwrap().to_string().contains("matches more than once"));
    assert_eq!(f.read("a"), b"x  y\nx  y\nx  y\n");
}

#[test]
fn line_edit_uses_the_requested_repeated_lines() {
    let f = Fixture::new(); f.write("a", "foo\nfoo\nfoo\n");
    let w = f.workspace();
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |_, _| true);
    let read = call(&mut tools, "workspace_read_lines", json::parse(r#"{"path":"a","start_line":1,"end_line":3}"#).unwrap());
    let read = json::parse(&read).unwrap();
    let tag = read.get("edit_tag").and_then(json::Value::as_str).unwrap();
    let args = json::parse(&format!(r#"{{"path":"a","operation":"replace","start_line":2,"end_line":3,"edit_tag":{},"new_text":"bar\n"}}"#, json::quote(tag))).unwrap();
    assert!(call(&mut tools, "workspace_edit_lines", args).starts_with("Applied "));
    assert_eq!(f.read("a"), b"foo\nbar\n");
}

#[test]
fn unicode_forms_and_case_aliases_are_refused_by_file_tools() {
    let f = Fixture::new(); f.write("ReadMe", "original\n");
    let w = f.workspace();
    for path in ["README", "\u{e9}.txt", "e\u{301}.txt"] {
        if path == "README" && !f.0.join(path).exists() { continue; }
        assert!(w.read(path).is_err());
        assert!(w.read_lines(path, 1, 1, 4096, &AtomicBool::new(false)).is_err());
        for body in [format!("*** Update File: {path}\n@@\n-original\n+changed"), format!("*** Add File: {path}\n+new"), format!("*** Update File: ReadMe\n*** Move to: {path}")] {
            assert!(edit::prepare(&w, &patch(&body)).is_err(), "{body}");
        }
    }
    assert_eq!(f.read("ReadMe"), b"original\n");
    assert_eq!(fs::read_dir(&f.0).unwrap().count(), 1);
}

#[test]
fn memory_changes_follow_the_documented_approval_policy() {
    // A memory note reaches every later chat, so with `ask` each change asks
    // first, `auto` shows it without waiting, and `never` refuses it without
    // asking (harness/docs/agent-tools.md, harness/docs/memory.md).
    let f = Fixture::new(); let w = f.workspace();
    let asked = AtomicUsize::new(0);
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Ask, |preview: vade_harness::coding_tools::Preview<'_>, _: &AtomicBool| {
        assert!(preview.approval_required, "{}", preview.title);
        asked.fetch_add(1, Ordering::SeqCst);
        true
    }).with_memory(true);
    assert_eq!(call(&mut tools, "memory_save", json::parse(r#"{"key":"style","text":"Use short answers."}"#).unwrap()), r#"{"entries":1}"#);
    assert_eq!(call(&mut tools, "memory_forget", json::parse(r#"{"key":"style"}"#).unwrap()), r#"{"entries":0}"#);
    drop(tools);
    assert_eq!(asked.load(Ordering::SeqCst), 2);
    // `auto` (`--yes`) shows the change and does not wait.
    let shown = AtomicUsize::new(0);
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Auto, |preview: vade_harness::coding_tools::Preview<'_>, _: &AtomicBool| {
        assert!(!preview.approval_required, "{}", preview.title);
        shown.fetch_add(1, Ordering::SeqCst);
        true
    }).with_memory(true);
    assert_eq!(call(&mut tools, "memory_save", json::parse(r#"{"key":"tone","text":"Plain words."}"#).unwrap()), r#"{"entries":1}"#);
    drop(tools);
    assert!(shown.load(Ordering::SeqCst) >= 1);
    let mut tools = CodingTools::new(&w, ApprovalPolicy::Never, |_, _| panic!("never asks no one")).with_memory(true);
    let arguments = json::parse(r#"{"key":"style","text":"Use short answers."}"#).unwrap();
    let call = ToolCall { id: "regression".into(), name: "memory_save".into(), arguments: json::stringify(&arguments).unwrap() };
    assert!(tools.execute(ToolInvocation { call: &call, arguments: &arguments, max_output_bytes: 4096 }, &AtomicBool::new(false)).is_err());
}

#[test]
#[ignore = "The credential name policy needs a maintainer decision"]
fn additional_credential_names_are_refused() {
    let f = Fixture::new();
    let paths = ["prod.env", "secrets.toml", "kubeconfig", "auth.json"];
    for path in paths { f.write(path, "SYNTHETIC_PRIVATE_CONTENT"); }
    let w = f.workspace();
    let accepted: Vec<_> = paths.into_iter().filter(|path| w.read(path).is_ok()).collect();
    assert!(accepted.is_empty(), "credential names were accepted: {accepted:?}");
}

#[test]
#[ignore = "New file mode overrides umask; the default needs a maintainer decision"]
fn new_file_respects_private_umask() {
    if std::env::var_os("VADE_EDIT_UMASK_CHILD").is_none() {
        let result = std::process::Command::new("sh")
            .args(["-c", "umask 077; exec \"$@\"", "sh"])
            .arg(std::env::current_exe().unwrap())
            .args(["--exact", "new_file_respects_private_umask", "--ignored", "--nocapture"])
            .env("VADE_EDIT_UMASK_CHILD", "1").output().unwrap();
        assert!(result.status.success(), "{}{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
        return;
    }
    use std::os::unix::fs::PermissionsExt;
    let f = Fixture::new();
    edit::prepare(&f.workspace(), &patch("*** Add File: new\n+synthetic"))
        .unwrap().apply(&AtomicBool::new(false)).unwrap();
    assert_eq!(fs::metadata(f.0.join("new")).unwrap().permissions().mode() & 0o777, 0o600);
}
