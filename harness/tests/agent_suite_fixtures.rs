//! The agent suite fixtures (tests/agent-suite) are complete in Git: every
//! input and fixture file that a task names exists and is tracked, and no file
//! under the suite is untracked or ignored. A fresh checkout runs the same tasks.
//! Every suite file also passes the release scan: under 5 MB, and no file type
//! that the scan blocks.
use std::{collections::BTreeSet, path::Path, process::Command};

const BLOCKED_TYPES: [&str; 5] = [".ptx", ".safetensors", ".log", ".pem", ".key"];
const MAX_FILE_BYTES: u64 = 5 * 1024 * 1024;

fn suite_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() { suite_files(&path, out) } else { out.push(path) }
    }
}

#[test]
fn every_suite_file_passes_the_release_scan() {
    let mut files = Vec::new();
    suite_files(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/agent-suite"), &mut files);
    for path in files {
        // The scan matches the type in any case.
        let name = path.file_name().unwrap().to_str().unwrap().to_ascii_lowercase();
        assert!(!BLOCKED_TYPES.iter().any(|t| name.ends_with(t)), "{}: the release scan blocks this file type", path.display());
        let bytes = path.metadata().unwrap().len();
        assert!(bytes <= MAX_FILE_BYTES, "{}: {bytes} bytes is over the 5 MB release limit", path.display());
    }
}

fn git_files(root: &Path, args: &[&str]) -> Option<BTreeSet<String>> {
    let out = Command::new("git").current_dir(root).args(args).args(["-z", "--", "tests/agent-suite"]).output().ok()?;
    out.status.success().then(|| String::from_utf8(out.stdout).unwrap().split('\0').filter(|s| !s.is_empty()).map(String::from).collect())
}

#[test]
fn every_named_task_input_exists_and_is_tracked() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let suite = root.join("tests/agent-suite/tasks");
    // A source export without .git can still check that the files exist.
    let tracked = git_files(root, &["ls-files", "--cached"]);
    if let Some(tracked) = &tracked {
        // Untracked and ignored files both count; Finder metadata does not.
        let stray: BTreeSet<_> = git_files(root, &["ls-files", "--others"]).unwrap().into_iter().filter(|p| !p.ends_with(".DS_Store")).collect();
        assert!(stray.is_empty(), "untracked or ignored suite files (git add them or remove them): {stray:?}");
        assert!(!tracked.is_empty(), "no tracked suite files");
    }
    let mut ids: Vec<_> = std::fs::read_dir(&suite).unwrap().map(|e| e.unwrap().path()).filter(|p| p.is_dir())
        .map(|p| p.file_name().unwrap().to_str().unwrap().to_string()).collect();
    ids.sort();
    assert_eq!(ids.len(), 12, "the suite README lists 12 tasks: {ids:?}");
    for id in &ids {
        let text = std::fs::read_to_string(suite.join(id).join("task.txt")).unwrap_or_else(|e| panic!("{id}/task.txt: {e}"));
        assert!(text.lines().any(|l| l == format!("id: {id}")), "{id}/task.txt must say id: {id}");
        let mut named = vec!["task.txt".to_string(), "solve.script".to_string()];
        let mut inputs = 0;
        for line in text.lines() {
            let words: Vec<&str> = line.split_whitespace().collect();
            match words.as_slice() {
                ["inputs:", paths @ ..] => { inputs += paths.len(); named.extend(paths.iter().map(|p| format!("workspace/{p}"))); }
                ["remove_before:", paths @ ..] => named.extend(paths.iter().map(|p| format!("workspace/{p}"))),
                ["request:", file] => named.push(file.to_string()),
                ["check:", "file-equals" | "json-equals", _, fixture] => named.push(fixture.to_string()),
                ["check:" | "step_check:", "run-with" | "fails-with", fixture, ..] => named.push(fixture.to_string()),
                _ => {}
            }
        }
        assert!(inputs > 0 || text.contains("mode: direct"), "{id}/task.txt names no inputs");
        for file in named {
            let rel = format!("tests/agent-suite/tasks/{id}/{file}");
            assert!(root.join(&rel).is_file(), "{rel} is named by {id}/task.txt but missing");
            if let Some(tracked) = &tracked { assert!(tracked.contains(&rel), "{rel} exists but is not tracked by Git"); }
        }
    }
}
