//! Explicit host hooks. Only native chat loads this configuration.
use std::{io::{self, Read, Write}, path::{Path, PathBuf}, process::{Child, Command, Stdio},
    sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}}, time::{Duration, Instant}};
use crate::{json::{self, Value}, tools::Workspace, user_files::{prefix, text}};

const CAP: usize = 4096;
const CONFIG_CAP: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Point { BeforeTool, AfterTool, OnStop }
impl Point {
    fn name(self) -> &'static str {
        match self { Self::BeforeTool => "before-tool", Self::AfterTool => "after-tool", Self::OnStop => "on-stop" }
    }
}
#[derive(Clone, Debug)]
struct Entry { point: Point, command: String, timeout: Duration }
#[derive(Clone, Debug, Default)]
pub struct Hooks { root: PathBuf, entries: Vec<Entry> }

impl Hooks {
    /// Read once at chat start. The caller reports errors and uses an empty set.
    pub fn load(workspace: &Workspace) -> io::Result<Self> {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            let dir = workspace.state_dir()?;
            match dir.workspace_file("hooks.conf")? {
                Some(file) => Self::parse(workspace.root(), &text(file, CONFIG_CAP)?),
                None => Ok(Self::default()),
            }
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        { let _ = workspace; Ok(Self::default()) }
    }
    fn parse(root: &Path, source: &str) -> io::Result<Self> {
        let invalid = || io::Error::other("invalid hooks.conf; see harness/docs/hooks.md");
        let config = json::parse(source)?;
        let Value::Object(fields) = &config else { return Err(invalid()); };
        if fields.len() != 1 || fields[0].0 != "hooks" { return Err(invalid()); }
        let entries = config.get("hooks").and_then(Value::as_array).ok_or_else(invalid)?;
        if entries.len() > 32 { return Err(invalid()); }
        let mut hooks = Self { root: root.into(), entries: Vec::new() };
        for entry in entries {
            let Value::Object(fields) = entry else { return Err(invalid()); };
            let mut names = std::collections::HashSet::new();
            if fields.iter().any(|(key, _)| !matches!(key.as_str(), "point" | "command" | "timeout_ms") || !names.insert(key)) { return Err(invalid()); }
            let point = match entry.get("point").and_then(Value::as_str) {
                Some("before-tool") => Point::BeforeTool, Some("after-tool") => Point::AfterTool,
                Some("on-stop") => Point::OnStop, _ => return Err(invalid()),
            };
            let command = entry.get("command").and_then(Value::as_str).ok_or_else(invalid)?;
            if command.trim().is_empty() || command.contains('\0') { return Err(invalid()); }
            let timeout = match entry.get("timeout_ms") {
                None => 30_000,
                Some(Value::Number(n)) => n.parse::<u64>().map_err(|_| invalid())?,
                _ => return Err(invalid()),
            };
            if !(1..=300_000).contains(&timeout) { return Err(invalid()); }
            hooks.entries.push(Entry { point, command: command.into(), timeout: Duration::from_millis(timeout) });
        }
        Ok(hooks)
    }
    /// Exit 2 blocks. Other failures are warnings and leave the tool allowed.
    pub fn before(&self, tool: &str, arguments: &Value, cancel: &AtomicBool, warn: &mut dyn FnMut(String)) -> Option<String> {
        if !self.entries.iter().any(|e| e.point == Point::BeforeTool) { return None; }
        let input = match input(tool, arguments, None) {
            Ok(input) => input,
            Err(_) => { warn("Hook before-tool warning: cannot encode input JSON.".into()); return None; }
        };
        for entry in self.entries.iter().filter(|e| e.point == Point::BeforeTool) {
            if cancel.load(Ordering::Relaxed) { break; }
            match run(&self.root, entry, tool, &input, cancel) {
                Ok(output) if output.code == Some(2) => {
                    let reason = if output.stderr.trim().is_empty() { "No reason supplied." } else { output.stderr.trim() };
                    return Some(format!("Hook blocked {tool}: {reason}"));
                }
                result => report(result, entry.point, warn),
            }
        }
        None
    }
    /// Append a bounded note without removing any part of the original receipt.
    pub fn after(&self, tool: &str, arguments: &Value, result: &mut String, limit: usize, cancel: &AtomicBool, warn: &mut dyn FnMut(String)) {
        if !self.entries.iter().any(|e| e.point == Point::AfterTool) { return; }
        let input = match input(tool, arguments, Some(prefix(result, CAP))) {
            Ok(input) => input,
            Err(_) => { warn("Hook after-tool warning: cannot encode input JSON.".into()); return; }
        };
        let mut notes = String::new();
        for entry in self.entries.iter().filter(|e| e.point == Point::AfterTool) {
            if cancel.load(Ordering::Relaxed) { break; }
            let output = run(&self.root, entry, tool, &input, cancel);
            if let Ok(output) = &output { notes.push_str(prefix(&output.stdout, CAP.saturating_sub(notes.len()))); }
            report(output, entry.point, warn);
        }
        if !notes.is_empty() {
            append_note(result, &notes, limit, warn);
        }
    }
    /// Called once after the native loop returns, including failure and cancel.
    pub fn stop(&self, show: &mut dyn FnMut(String)) {
        let cancel = AtomicBool::new(false);
        for entry in self.entries.iter().filter(|e| e.point == Point::OnStop) {
            let result = run(&self.root, entry, "", "{}", &cancel);
            if let Ok(output) = &result && !output.stdout.is_empty() { show(format!("[on-stop hook]\n{}", output.stdout)); }
            report(result, entry.point, show);
        }
    }
}

fn append_note(result: &mut String, notes: &str, limit: usize, warn: &mut dyn FnMut(String)) {
    let budget = limit.saturating_sub(result.len()).min(CAP);
    let label = "[after-tool hook note]\n";
    // Keep JSON receipts parseable. Error and uncertainty fields still govern
    // recovery, retry, compaction, and fork checks after the note is saved.
    let object = match json::parse(result) { Ok(Value::Object(fields)) => Some(fields), _ => None };
    let mut key = "vade_hook_note".to_owned();
    if let Some(fields) = &object {
        while fields.iter().any(|(name, _)| name == &key) { key.push('_'); }
    }
    let mut end = notes.len().min(budget);
    loop {
        let kept = prefix(notes, end);
        if kept.is_empty() { warn("Hook note was omitted to fit the tool result limit.".into()); return; }
        let note = match &object {
            Some(fields) => format!("{}{}:{}", if fields.is_empty() { "" } else { "," }, json::quote(&key), json::quote(&format!("{label}{kept}"))),
            None => format!("\n{label}{kept}"),
        };
        if note.len() <= budget {
            if kept.len() < notes.len() { warn("Hook note was cut to fit the tool result limit.".into()); }
            if object.is_some() { result.insert_str(result.rfind('}').expect("parsed JSON object"), &note); }
            else { result.push_str(&note); }
            return;
        }
        end = kept.len().saturating_sub(note.len() - budget);
    }
}

fn input(tool: &str, arguments: &Value, summary: Option<&str>) -> io::Result<String> {
    let mut fields = vec![("tool".into(), Value::String(tool.into())), ("arguments".into(), arguments.clone())];
    if let Some(summary) = summary { fields.push(("result_summary".into(), Value::String(summary.into()))); }
    json::stringify(&Value::Object(fields))
}
struct Output { code: Option<i32>, stdout: String, stderr: String }
fn report(result: io::Result<Output>, point: Point, warn: &mut dyn FnMut(String)) {
    match result {
        Ok(output) if output.code == Some(0) => (),
        Ok(output) => warn(format!("Hook {} warning: exit {}. {}", point.name(), output.code.map_or_else(|| "signal".into(), |c| c.to_string()), output.stderr)),
        Err(error) => warn(format!("Hook {} warning: {error}", point.name())),
    }
}

// The standard library cannot signal a process group. The hook leads its
// own group; stop it with the guarded group stop. Drop also covers pipe and
// wait errors.
struct Owned(Child);
impl Drop for Owned {
    fn drop(&mut self) {
        crate::process_group::kill_group(&mut self.0);
        let _ = self.0.wait();
    }
}
#[derive(Default)]
struct Capture { bytes: Vec<u8>, done: bool, failed: bool, cut: bool }
fn drain(mut pipe: impl Read + Send + 'static) -> Arc<Mutex<Capture>> {
    let capture = Arc::new(Mutex::new(Capture::default()));
    let shared = Arc::clone(&capture);
    std::thread::spawn(move || {
        let mut buf = [0; 4096];
        loop {
            let read = pipe.read(&mut buf);
            if matches!(&read, Err(e) if e.kind() == io::ErrorKind::Interrupted) { continue; }
            let mut capture = shared.lock().unwrap_or_else(|e| e.into_inner());
            match read {
                Ok(0) => { capture.done = true; break; }
                Ok(n) => {
                    let kept = n.min(CAP.saturating_sub(capture.bytes.len()));
                    capture.bytes.extend_from_slice(&buf[..kept]);
                    capture.cut |= kept < n;
                }
                Err(_) => { capture.failed = true; capture.done = true; break; }
            }
        }
    });
    capture
}
fn captured(capture: &Mutex<Capture>) -> String {
    let capture = capture.lock().unwrap_or_else(|e| e.into_inner());
    let value = String::from_utf8_lossy(&capture.bytes);
    let marker = if capture.failed || !capture.done { "\n[hook output incomplete]" } else { "\n[hook output cut]" };
    if capture.failed || !capture.done || capture.cut || value.len() > CAP {
        format!("{}{marker}", prefix(&value, CAP - marker.len()))
    } else { value.into_owned() }
}
fn drained(capture: &Mutex<Capture>) -> bool {
    capture.lock().unwrap_or_else(|e| e.into_inner()).done
}
fn run(root: &Path, entry: &Entry, tool: &str, input: &str, cancel: &AtomicBool) -> io::Result<Output> {
    if cancel.load(Ordering::Relaxed) { return Err(io::Error::other("hook cancelled")); }
    let mut command = Command::new("/bin/sh");
    command.args(["-c", &entry.command]).current_dir(root)
        .env("VADE_HOOK_POINT", entry.point.name()).env("VADE_TOOL", tool).env("VADE_PROJECT", root)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(unix)]
    { use std::os::unix::process::CommandExt; command.process_group(0); }
    let start = Instant::now();
    let mut child = Owned(command.spawn()?);
    let stdout = drain(child.0.stdout.take().expect("hook stdout"));
    let stderr = drain(child.0.stderr.take().expect("hook stderr"));
    let mut stdin = child.0.stdin.take().expect("hook stdin");
    let input = input.as_bytes().to_vec();
    std::thread::spawn(move || { let _ = stdin.write_all(&input); });
    loop {
        if cancel.load(Ordering::Relaxed) { return Err(io::Error::other("hook cancelled")); }
        if let Some(status) = child.0.try_wait()? {
            // Keep the shell's decision even if a descendant holds a pipe open.
            // Stop the remaining group before waiting for the readers to finish.
            drop(child);
            let drain_start = Instant::now();
            while !(drained(&stdout) && drained(&stderr)) && drain_start.elapsed() < Duration::from_secs(1) {
                std::thread::sleep(Duration::from_millis(5));
            }
            return Ok(Output { code: status.code(), stdout: captured(&stdout), stderr: captured(&stderr) });
        }
        if start.elapsed() >= entry.timeout { return Err(io::Error::new(io::ErrorKind::TimedOut, "hook timed out")); }
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[cfg(test)]
mod tests;
