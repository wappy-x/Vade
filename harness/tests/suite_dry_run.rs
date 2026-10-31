//! `vade-harness suite --dry-run` over the real agent suite: every task passes
//! with its solve.script and fails with its wrong.script, through the real
//! tool executor and no model. Small suites in temporary Git checkouts prove
//! that the gate fails when a script does not do what it must.
#[path = "support/home.rs"]
mod home;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn suite(args: &[&str], cwd: &Path) -> Output { suite_in(args, cwd, None) }

/// `tmp` becomes the child's TMPDIR: a live run's fixed workspace folder then
/// belongs to this test, not to another live run or test on the same host.
fn suite_in(args: &[&str], cwd: &Path, tmp: Option<&Path>) -> Output {
    // The fixture HOME hides the developer's rustup proxies, so name the toolchain that builds this test.
    let cargo = PathBuf::from(env!("CARGO"));
    let toolchain = cargo.parent().and_then(Path::parent).filter(|t| t.join("bin/rustc").is_file() && t.join("lib").is_dir());
    let mut command = home::harness();
    command.arg("suite").args(args).current_dir(cwd);
    if let Some(t) = toolchain { command.arg("--toolchain").arg(t); }
    if let Some(t) = tmp { command.env("TMPDIR", t); }
    command.output().unwrap()
}

fn text(out: &Output) -> String { format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)) }

#[test]
fn the_agent_suite_passes_its_dry_run_gate() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    // A source export without .git cannot prove which files a fresh checkout has.
    if !Command::new("git").current_dir(root).args(["rev-parse", "--git-dir"]).output().is_ok_and(|o| o.status.success()) { eprintln!("skipped: the suite runner needs a Git checkout"); return; }
    let out = suite(&["--dry-run"], root);
    let all = text(&out);
    assert!(out.status.success(), "{all}");
    let tasks = std::fs::read_dir(root.join("tests/agent-suite/tasks")).unwrap().count();
    let last = all.lines().rfind(|l| l.starts_with("DRY RUN ")).unwrap_or_default();
    assert!(last.starts_with("DRY RUN PASS: "), "{all}");
    assert!(last.contains(&format!(" of {tasks} tasks ")), "{last}");
    assert_eq!(all.lines().filter(|l| l.ends_with("| OK") || l.contains("| SKIP: ")).count(), tasks, "{all}");
}

/// A Git checkout with one suite task made of the given files.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str, files: &[(&str, &str)]) -> Self {
        let dir = std::env::temp_dir().join(format!("vade-suite-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for (path, body) in files {
            let p = dir.join("suite/tasks/t").join(path);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        }
        std::fs::write(dir.join("suite/VERSION"), "test-suite\n").unwrap();
        let git = |args: &[&str]| assert!(Command::new("git").current_dir(&dir).args(args).output().unwrap().status.success(), "git {args:?}");
        git(&["init", "-q"]);
        git(&["add", "-A"]);
        Scratch(dir)
    }
    fn run(&self) -> (bool, String) {
        let out = suite(&["--dry-run", "--suite", "suite"], &self.0);
        assert!(matches!(out.status.code(), Some(0 | 1)), "{}", text(&out));
        (out.status.success(), text(&out))
    }
}

impl Drop for Scratch { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

const TASK: &str = "id: t\ntitle: Write one file\ncategory: test\ninputs: in.txt\nbudget_seconds: 60\n\nstep: new\nprompt:\n  Copy in.txt to out.txt.\n\ncheck: last-exit-zero\ncheck: file-equals out.txt in.txt.want\ncheck: unchanged in.txt\n";
const RUBRIC: &str = "may_change: out.txt\nmax_changed_lines: 2\nanswer_mentions: Done\nanswer_max_chars: 100\n";
const WRITE_OK: &str = ">>> patch\n*** Begin Patch\n*** Add File: out.txt\n+alpha\n*** End Patch\n>>> answer\nDone.\n";
const WRITE_BAD: &str = ">>> patch\n*** Begin Patch\n*** Add File: out.txt\n+beta\n*** End Patch\n>>> answer\nDone.\n";

fn files<'a>(solve: &'a str, wrong: &'a str) -> Vec<(&'a str, &'a str)> {
    vec![("task.txt", TASK), ("workspace/in.txt", "alpha\n"), ("in.txt.want", "alpha\n"), ("rubric.txt", RUBRIC), ("solve.script", solve), ("wrong.script", wrong)]
}

#[test]
fn the_gate_passes_a_good_task() {
    let (ok, all) = Scratch::new("good", &files(WRITE_OK, WRITE_BAD)).run();
    assert!(ok, "{all}");
    assert!(all.contains("test-suite dry run | t | test | solve PASS 3/3 quality 3/3 (1 changed lines) | wrong FAIL 2/3 quality 0/3 (file-equals out.txt in.txt.want -> line 1: got \"beta\", want \"alpha\") | OK"), "{all}");
    assert!(all.contains("DRY RUN PASS: 1 of 1 tasks"), "{all}");
}

#[test]
fn the_gate_fails_a_solve_that_fails_or_a_wrong_that_passes() {
    let (ok, all) = Scratch::new("bad-solve", &files(WRITE_BAD, WRITE_BAD)).run();
    assert!(!ok, "{all}");
    assert!(all.contains("GATE FAIL: solve.script must pass"), "{all}");
    assert!(all.contains("DRY RUN FAIL: 0 of 1 tasks"), "{all}");
    let (ok, all) = Scratch::new("good-wrong", &files(WRITE_OK, WRITE_OK)).run();
    assert!(!ok, "{all}");
    assert!(all.contains("GATE FAIL: wrong.script must fail a check, got PASS"), "{all}");
}

#[test]
fn the_gate_fails_unused_replies_and_a_missing_wrong_script() {
    let extra = format!("{WRITE_OK}>>> answer\nNever asked for.\n");
    let (ok, all) = Scratch::new("extra", &files(&extra, WRITE_BAD)).run();
    assert!(!ok, "{all}");
    assert!(all.contains("solve.script has 1 replies the harness never asked for"), "{all}");
    let mut no_wrong = files(WRITE_OK, WRITE_BAD);
    no_wrong.pop();
    let mut no_rubric = files(WRITE_OK, WRITE_BAD);
    no_rubric.retain(|(p, _)| *p != "rubric.txt");
    let s = Scratch::new("no-rubric", &no_rubric);
    let out = suite(&["--dry-run", "--suite", "suite", "--out", "results"], &s.0);
    assert_eq!(out.status.code(), Some(2), "{}", text(&out));
    assert!(text(&out).contains("t/rubric.txt: No such file"), "{}", text(&out));
    assert!(!s.0.join("results").exists(), "a refusal made the results folder");
    // A solve that passes every check but edits a file the rubric does not allow fails the gate.
    let sloppy = format!(">>> patch\n*** Begin Patch\n*** Add File: notes.txt\n+x\n*** End Patch\n{WRITE_OK}");
    let (ok, all) = Scratch::new("sloppy", &files(&sloppy, WRITE_BAD)).run();
    assert!(!ok, "{all}");
    assert!(all.contains("solve.script must score 3/3 on rubric.txt, got 2: collateral edit: notes.txt"), "{all}");
    let (ok, all) = Scratch::new("no-wrong", &no_wrong).run();
    assert!(!ok, "{all}");
    assert!(all.contains("wrong ERROR | GATE FAIL: wrong.script:"), "{all}");
}

#[test]
fn the_runner_refuses_unknown_checks_and_untracked_files() {
    let bad = TASK.replace("check: unchanged in.txt", "check: max-turns 6");
    let mut f = files(WRITE_OK, WRITE_BAD);
    f[0].1 = &bad;
    let s = Scratch::new("unknown-check", &f);
    let out = suite(&["--dry-run", "--suite", "suite", "--out", "results"], &s.0);
    assert_eq!(out.status.code(), Some(2), "{}", text(&out));
    assert!(text(&out).contains("t/task.txt: unknown check max-turns 6"), "{}", text(&out));
    assert!(!s.0.join("results").exists(), "a refusal made the results folder");
    let s = Scratch::new("untracked", &files(WRITE_OK, WRITE_BAD));
    std::fs::write(s.0.join("suite/tasks/t/stray.txt"), "x").unwrap();
    let out = suite(&["--dry-run", "--suite", "suite", "--out", "results"], &s.0);
    assert_eq!(out.status.code(), Some(2), "{}", text(&out));
    assert!(text(&out).contains("untracked or ignored files in the suite"), "{}", text(&out));
    assert!(!s.0.join("results").exists(), "a refusal made the results folder");
}

/// A loopback model that streams fixed replies as SSE with a `token_id` on
/// every chunk, in a cycle. `change` gives a token ID offset per answered call
/// number (1-based, over all runs), to make one call of one run differ.
fn stub_model(replies: Vec<String>, change: Option<(usize, u64)>) -> String {
    use std::io::{BufRead, BufReader, Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1/chat/completions", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let mut served = 0;
        for stream in listener.incoming().flatten() {
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let (mut length, mut first) = (0, String::new());
            let _ = reader.read_line(&mut first);
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() { break; }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") { length = v.trim().parse().unwrap_or(0); }
            }
            let mut body = vec![0; length];
            let _ = reader.read_exact(&mut body);
            let mut out = stream;
            // Only chat calls take a reply; the harness may probe other paths first.
            if !first.starts_with("POST ") || !first.contains("/chat/completions") {
                let _ = write!(out, "HTTP/1.1 404 Not Found\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}");
                continue;
            }
            let delta = &replies[served % replies.len()];
            served += 1;
            let offset = change.filter(|(call, _)| *call == served).map_or(0, |(_, by)| by);
            let _ = write!(out, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
            let _ = write!(out, "data: {{\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":null,\"token_id\":{}}}]}}\n\n", 7 + offset);
            let finish = if delta.contains("tool_calls") { "tool_calls" } else { "stop" };
            let _ = write!(out, "data: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{finish}\",\"token_id\":8}}],\"usage\":{{\"prompt_tokens\":100,\"completion_tokens\":2}}}}\n\ndata: [DONE]\n\n");
        }
    });
    url
}

fn call(n: usize, name: &str, args: &str) -> String {
    let quoted: String = args.chars().flat_map(|c| match c { '"' => vec!['\\', '"'], '\\' => vec!['\\', '\\'], '\n' => vec!['\\', 'n'], c => vec![c] }).collect();
    format!("{{\"role\":\"assistant\",\"tool_calls\":[{{\"index\":0,\"id\":\"call_{n}\",\"type\":\"function\",\"function\":{{\"name\":\"{name}\",\"arguments\":\"{quoted}\"}}}}]}}")
}

#[test]
fn a_live_run_compares_token_ids_and_grades() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    if !Command::new("git").current_dir(root).args(["rev-parse", "--git-dir"]).output().is_ok_and(|o| o.status.success()) { eprintln!("skipped: the suite runner needs a Git checkout"); return; }
    let replies = vec![
        call(1, "workspace_read", r#"{"path":"retention_policy.docx"}"#),
        call(2, "workspace_read", r#"{"path":"QUESTIONS.md"}"#),
        call(3, "apply_patch", "{\"patch\":\"*** Begin Patch\\n*** Add File: ANSWERS.txt\\n+Q1: 90 days\\n+Q2: zstd-3\\n+Q3: 512 MiB\\n+Q4: us-central-spark\\n+Q5: sec-ops-lead\\n*** End Patch\\n\"}"),
        "{\"role\":\"assistant\",\"content\":\"ANSWERS.txt holds the five answers.\"}".to_string(),
    ];
    let out = std::env::temp_dir().join(format!("vade-suite-live-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    // The child's own TMPDIR holds the fixed workspace folder; it must be gone after every run.
    // It is outside the Vade checkout, as in a real live run, so no outer AGENTS.md is in the prompt.
    let tmp = std::env::temp_dir().join(format!("vade-suite-live-tmp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let fixed = tmp.join("vade-harness-suite-live");
    let live_args = |url: &str, runs: &str, dir: &Path| -> Output {
        let out = suite_in(&["--live", "--endpoint", url, "--model", "stub", "--runs", runs, "--only", "docx-policy-qa", "--out", dir.to_str().unwrap()], root, Some(&tmp));
        assert!(!fixed.exists(), "the live workspace folder is left after the run: {}", text(&out));
        out
    };
    let live = |url: &str, dir: &Path| live_args(url, "3", dir);
    // A folder left by another run: refused (exit 2), no --out folder is made and no connection to the endpoint.
    let quiet = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    quiet.set_nonblocking(true).unwrap();
    let quiet_url = format!("http://{}/v1/chat/completions", quiet.local_addr().unwrap());
    std::fs::create_dir(&fixed).unwrap();
    let refused = suite_in(&["--live", "--endpoint", &quiet_url, "--model", "stub", "--only", "docx-policy-qa", "--out", out.join("refused").to_str().unwrap()], root, Some(&tmp));
    assert_eq!(refused.status.code(), Some(2), "{}", text(&refused));
    assert!(text(&refused).contains("another live suite run uses it"), "{}", text(&refused));
    assert!(!out.join("refused").exists(), "a refusal made the --out folder");
    std::fs::remove_dir(&fixed).unwrap();
    // A TMPDIR that does not exist: refused with the real cause, not "another live suite run".
    let missing = tmp.join("missing");
    let refused = suite_in(&["--live", "--endpoint", &quiet_url, "--model", "stub", "--only", "docx-policy-qa", "--out", out.join("refused").to_str().unwrap()], root, Some(&missing));
    assert_eq!(refused.status.code(), Some(2), "{}", text(&refused));
    assert!(!text(&refused).contains("another live suite run uses it"), "{}", text(&refused));
    assert!(!out.join("refused").exists(), "a refusal made the --out folder");
    // The child has exited, so a connection that it made would wait in the backlog.
    assert_eq!(quiet.accept().map(|_| ()).map_err(|e| e.kind()), Err(std::io::ErrorKind::WouldBlock), "a refused live run connected to the endpoint");
    // The same replies in every run: every run passes, the IDs match, grade A.
    let run = live(&stub_model(replies.clone(), None), &out.join("same"));
    let all = text(&run);
    assert!(run.status.success(), "{all}");
    assert!(all.contains("| agent-suite-2 | stub | 1 tasks | 3 runs | 3/3 task runs pass | quality 3.00 | 1/1 tasks exact | grade A |"), "{all}");
    assert!(all.contains("| docx-policy-qa | document-reading | 3/3 | 3 3 3 | yes | PASS PASS PASS |"), "{all}");
    assert!(all.contains("| document-reading | 1 | 3/3 | 3.00 |"), "{all}");
    assert!(all.contains("## Failed task runs\n\nNone."), "{all}");
    for k in 1..=3 { assert!(out.join(format!("same/run-{k}/card.json")).is_file()); }
    let card = std::fs::read_to_string(out.join("same/card.json")).unwrap();
    assert!(card.contains("\"grade\": \"A\""), "{card}");
    // One token of the second model call of run 3 differs: still a pass, not exact.
    let run = live(&stub_model(replies, Some((4 * 2 + 2, 1))), &out.join("differ"));
    let all = text(&run);
    assert!(run.status.success(), "{all}");
    assert!(all.contains("| 3/3 task runs pass | quality 3.00 | 0/1 tasks exact | grade F |"), "{all}");
    assert!(all.contains("| docx-policy-qa | document-reading | 3/3 | 3 3 3 | no (model call 2 differs with the same request) | PASS PASS PASS |"), "{all}");
    // A run that misreads a path first and then writes a wrong answer: the
    // card names that call as the first wrong tool call.
    let bad = vec![
        call(1, "workspace_read", r#"{"path":"policy.docx"}"#),
        call(2, "apply_patch", "{\"patch\":\"*** Begin Patch\\n*** Add File: ANSWERS.txt\\n+Q1: 30 days\\n*** End Patch\\n\"}"),
        "{\"role\":\"assistant\",\"content\":\"Done.\"}".to_string(),
    ];
    let dir = out.join("bad");
    let run = live_args(&stub_model(bad, None), "1", &dir);
    let all = text(&run);
    assert_eq!(run.status.code(), Some(1), "{all}");
    assert!(all.contains("| 0/1 task runs pass | quality 0.00 | 0/1 tasks exact | grade none (needs 3 runs) |"), "{all}");
    assert!(all.contains("- FAIL docx-policy-qa (run 1): 3/4 checks; first failure: file-equals ANSWERS.txt expected/ANSWERS.txt -> line 1: got \"Q1: 30 days\", want \"Q1: 90 days\"; first wrong tool call: call 1 workspace_read {\"path\":\"policy.docx\"} -> Read refused"), "{all}");
    let card = std::fs::read_to_string(dir.join("run-1/card.md")).unwrap();
    assert!(card.contains("| document-reading | 1 | 0/1 | 0.00 |"), "{card}");
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&tmp);
}
