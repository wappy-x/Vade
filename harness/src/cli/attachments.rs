use std::path::{Component, Path, PathBuf};
use vade_harness::{session::MAX_PROMPT_BYTES, tools::Workspace};

const HEADER: &str = "\n\n[attached files]\n";

// Decode terminal drag paths without invoking a shell or expanding variables.
fn path_token(text: &str) -> Result<(String, usize), String> {
    let (mut path, mut quote, mut escaped) = (String::new(), None, false);
    for (at, c) in text.char_indices() {
        if escaped { path.push(c); escaped = false; continue; }
        if c == '\\' && quote != Some('\'') { escaped = true; continue; }
        if Some(c) == quote { quote = None; continue; }
        if quote.is_none() && matches!(c, '\'' | '"') { quote = Some(c); continue; }
        if quote.is_none() && c.is_whitespace() { return Ok((path, at)); }
        path.push(c);
    }
    if escaped || quote.is_some() { return Err("Attachment path has an unfinished quote or escape; nothing sent.".into()); }
    Ok((path, text.len()))
}

pub(crate) fn absolute_input(text: &str) -> bool {
    path_token(text.trim()).is_ok_and(|(path, end)| path.starts_with('/') && end == text.trim().len())
}

fn existing(workspace: &Workspace, selected_root: &Path, token: &str) -> Option<(String, u64)> {
    let token = token.trim_end_matches(['.', ',', ';', ':', ')']);
    let path = Path::new(token.rsplit_once(':').filter(|(_, line)| line.parse::<u32>().is_ok()).map_or(token, |(name, _)| name));
    let selected = if path.is_absolute() { std::env::current_dir().map(|cwd| cwd.join(selected_root)).unwrap_or_else(|_| workspace.root().to_owned()) } else { PathBuf::new() };
    let relative = if path.is_absolute() { path.strip_prefix(workspace.root()).or_else(|_| path.strip_prefix(&selected)).ok()? } else { path };
    let mut name = PathBuf::new();
    for part in relative.components() {
        match part { Component::Normal(p) => name.push(p), Component::CurDir => (), Component::ParentDir if name.pop() => (), _ => return None }
    }
    let mut exact = workspace.root().to_owned();
    for part in name.components() {
        if !std::fs::read_dir(&exact).ok()?.filter_map(Result::ok).any(|entry| entry.file_name() == part.as_os_str()) { return None; }
        exact.push(part);
    }
    let meta = std::fs::symlink_metadata(workspace.root().join(&name)).ok()?;
    if !meta.is_file() && !(meta.file_type().is_symlink() && std::fs::metadata(workspace.root().join(&name)).is_ok_and(|m| m.is_file())) { return None; }
    Some((name.to_str()?.to_owned(), meta.len()))
}

pub(crate) fn resolves_absolute(workspace: &Workspace, selected_root: &Path, text: &str) -> bool {
    path_token(text.trim()).is_ok_and(|(path, _)| path.starts_with('/') && existing(workspace, selected_root, &path).is_some())
}

pub(crate) fn display(text: &str) -> &str { text.split(HEADER).next().unwrap_or(text) }

pub(crate) fn complete(workspace: &Workspace, input: &str, cursor: usize) -> Option<(String, String)> {
    if cursor != input.len() || input.ends_with(char::is_whitespace) { return None; }
    let token = input.split_whitespace().next_back()?.strip_prefix('@')?;
    if token.starts_with('/') || token.contains(['\\', '\'', '"']) { return Some((String::new(), String::new())); }
    let (directory, prefix) = token.rsplit_once('/').unwrap_or((".", token));
    let names = workspace.list(directory).unwrap_or_default();
    let matches: Vec<_> = names.iter().filter(|name| name.as_str() != "[listing limit reached]" && name.starts_with(prefix)).collect();
    let candidates = matches.iter().take(10).map(|name| name.as_str()).collect::<Vec<_>>().join("\n");
    let Some(first) = matches.first() else { return Some((String::new(), candidates)); };
    let mut name = first.as_str();
    for other in &matches { while !other.starts_with(name) { name = &name[..name.len() - 1]; } }
    let mut suffix = String::new();
    for c in name[prefix.len()..].chars() {
        if c.is_whitespace() || matches!(c, '\\' | '\'' | '"') { suffix.push('\\'); }
        suffix.push(c);
    }
    let full = if directory == "." { name.to_string() } else { format!("{directory}/{name}") };
    if matches.len() == 1 && workspace.list(&full).is_ok() { suffix.push('/'); }
    Some((suffix, candidates))
}

pub(crate) fn prepare(workspace: &Workspace, selected_root: &Path, input: &str) -> Result<(String, Vec<String>), String> {
    let (mut paths, mut at) = (Vec::new(), 0);
    let absolute = absolute_input(input) && !input.trim_start().starts_with("//");
    if input.trim_start().starts_with("//") { return Ok((input.to_owned(), Vec::new())); }
    while at < input.len() {
        let c = input[at..].chars().next().unwrap();
        let boundary = at == 0 || input[..at].chars().next_back().is_some_and(|c| c.is_whitespace() || c == '(');
        let rest = &input[at..];
        let bare = absolute && (rest.starts_with('/') || rest.starts_with("\"/") || rest.starts_with("'/"));
        if boundary && (c == '@' || bare) {
            let Ok((path, end)) = path_token(if c == '@' { &rest[1..] } else { rest }) else { at += c.len_utf8(); continue; };
            if let Some(file) = existing(workspace, selected_root, &path) { if !paths.contains(&file) { paths.push(file); } }
            at += end + usize::from(c == '@');
        } else { at += c.len_utf8(); }
    }
    let (mut sent, mut chips) = (input.to_owned(), Vec::new());
    for (name, bytes) in paths {
        let path = Path::new(&name);
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
        if ["pdf", "png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff", "svg", "heic"].contains(&extension.as_str()) {
            return Err(format!("{name}: images cannot be attached; PDF text extraction is not built yet; nothing sent."));
        }
        if bytes > MAX_PROMPT_BYTES as u64 { return Err(format!("{name} is {bytes} bytes; attachments are capped at 64 KiB per message. Ask Vade to read it instead; nothing sent.")); }
        let text = workspace.read(&name).map_err(|_| format!("{name}: cannot attach this file (requires readable UTF-8 text, visible ASCII names and no symlinks); nothing sent."))?;
        if text.chars().any(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t')) {
            return Err(format!("Attachment {name}: binary/control bytes are not text; nothing sent."));
        }
        let chip = format!("[file: {name}, {} lines]", text.lines().count());
        if chips.is_empty() { sent.push_str(HEADER); }
        sent.push_str(&format!("{chip}\n{text}\n"));
        if sent.len() > MAX_PROMPT_BYTES { return Err("Attachments exceed the 64 KiB message limit; nothing sent.".into()); }
        chips.push(chip);
    }
    Ok((sent, chips))
}

impl super::App {
    pub(crate) fn internal_prompt(&self, text: &str) -> bool {
        self.answers.continuing || text == vade_harness::agent::CONTINUATION_PROMPT || text == vade_harness::agent::TOOL_CONTINUATION_PROMPT
    }
    pub(crate) fn keep_draft(&mut self, text: &str) -> &'static str {
        if text.trim().is_empty() || self.internal_prompt(text) { return ""; }
        let draft = display(text);
        if self.terminal.raw() && self.input.is_empty() {
            self.input.set(draft.to_owned()); " Draft kept in the editor."
        } else {
            self.reset_recall();
            self.input.remember(draft);
            if self.terminal.raw() { " Draft kept in history (Up)." } else { "" }
        }
    }
    pub(crate) fn prepare_input(&mut self, input: String) -> Result<String, String> {
        match prepare(&self.workspace, &self.config.workspace, &input) {
            Ok((sent, _)) => Ok(sent),
            Err(reason) => { self.failed = true; let kept = self.keep_draft(&input); Err(format!("{reason}{kept} Wrap @name in backticks to send it as text.")) }
        }
    }
    pub(crate) fn start_input(&mut self, input: String) -> std::io::Result<()> {
        match self.prepare_input(input) {
            Ok(sent) => self.start(sent),
            Err(reason) => { self.failed = true; self.notice(&reason) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompts_files_drafts_recall_and_completion() {
        let root = std::env::temp_dir().join(format!("vade-attach-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let outside = root.with_extension("outside"); std::fs::write(&outside, "outside workspace").unwrap();
        for (name, bytes) in [("LICENSE", &b"license"[..]), ("NOTICE", b"notice"), ("input.txt", b"private file bytes\n"), ("input-two.txt", b"two"), ("space name.txt", b"space"), ("binary", b"\0"), (".hidden", b"hidden"), ("image.png", b"image"), ("file.pdf", b"pdf")] { std::fs::write(root.join(name), bytes).unwrap(); }
        std::fs::write(root.join("big"), vec![b'x'; MAX_PROMPT_BYTES + 1]).unwrap();
        std::fs::write(root.join("invalid"), b"\xff").unwrap(); for name in ["half-a", "half-b"] { std::fs::write(root.join(name), vec![b'x'; 40 * 1024]).unwrap(); }
        std::fs::create_dir(root.join("dir")).unwrap(); std::fs::write(root.join("dir/child"), "nested").unwrap();
        #[cfg(unix)] std::os::unix::fs::symlink("input.txt", root.join("link")).unwrap();
        let workspace = Workspace::new(&root).unwrap();
        // Keep the absolute input unquoted when the temporary parent contains spaces.
        let bare_path = root.join("input.txt").display().to_string().replace(' ', "\\ ");
        let ordinary = ["/// @notice Returns the balance", " * @license MIT", "@INPUT.TXT", "@DIR/child", "@dir/CHILD", "explain what /etc/hosts does", "what is 6 / 2", "the endpoint /v1/chat/completions 404s", "cat /dev/zero", "File \"/usr/lib/example.py\", line 5", "@Override void f() {}", "```python\n@dataclass\nclass C: pass\n```", "npm i @types/node", "meet me @ 5pm", "@john's idea", "@", "@missing", "@missing.png", "@\"input.txt", "@input.txt\\", "//etc/hosts", "// comment", "/* comment */", "run /help", "@dir", "@../missing", "https://example.invalid/api", "plain prompt"];
        for input in ordinary.iter().map(|s| s.to_string()).chain([format!("{}:1:5: error", root.join("input.txt").display()), format!("log {} loaded", root.join("input.txt").display()), format!("person{}mail.invalid", '@'), format!("explain @{}", outside.display())]) { let (sent, chips) = prepare(&workspace, &root, &input).unwrap(); assert_eq!(sent, input); assert!(chips.is_empty()); }
        for input in ["@LICENSE", "@NOTICE", "@dir/child", "(@input.txt)", "@input.txt:12", "@john's idea about @input.txt", "look at @input.txt,", "@input.txt. @input.txt; @input.txt: @input.txt)", "@\"space name.txt\"", "@space\\ name.txt", "@input.txt\u{3000}", "@./input.txt"].into_iter().map(str::to_string).chain([format!("'{}'", root.join("space name.txt").display()), format!("  {bare_path}  ")]) { let (sent, chips) = prepare(&workspace, &root, &input).unwrap(); assert_eq!(chips.len(), 1); assert_eq!(display(&sent), input); }
        for input in ["@binary", "@invalid", "@big", "@.hidden", "@image.png", "@file.pdf", "@link", "@input.txt @big", "@half-a @half-b"] { assert!(prepare(&workspace, &root, input).unwrap_err().contains("nothing sent")); }
        assert!(!resolves_absolute(&workspace, &root, "/hlep"));
        assert!(resolves_absolute(&workspace, &root, &bare_path));
        assert!(complete(&workspace, "@inp ", 5).is_none()); assert!(complete(&workspace, "@inp\n", 5).is_none()); assert!(complete(&workspace, "@inp", 2).is_none()); assert_eq!(complete(&workspace, "@/inp", 5).unwrap(), (String::new(), String::new()));
        assert_eq!(complete(&workspace, "@inp", 4).unwrap().0, "ut");
        assert_eq!(complete(&workspace, "@input.", 7).unwrap().0, "txt");
        assert_eq!(complete(&workspace, "@dir", 4).unwrap().0, "/");
        let (suffix, candidates) = complete(&workspace, "@input", 6).unwrap(); assert!(suffix.is_empty()); assert!(candidates.contains("input-two.txt") && candidates.contains("input.txt"));
        let super::super::args::Parsed::Run(config) = super::super::args::Config::parse(["demo".to_owned(), "--workspace".into(), root.display().to_string()]).unwrap() else { panic!() };
        let mut app = super::super::App::new(config).unwrap(); app.input.load_history(&std::sync::Arc::new(Workspace::new(&root).unwrap())).unwrap();
        app.input.set("@inp".into()); app.key(vade_harness::terminal::Key::Tab).unwrap(); assert_eq!(app.input.text(), "@input"); assert!(app.attachment_completion.1.is_empty()); app.key(vade_harness::terminal::Key::Tab).unwrap(); assert!(app.attachment_completion.1.contains("input-two.txt"));
        app.input.set("@inp ".into()); app.key(vade_harness::terminal::Key::Tab).unwrap(); assert_eq!(app.input.text(), "@inp     ");
        assert!(app.prepare_input("@binary".into()).unwrap_err().contains("nothing sent")); assert_eq!(app.input.text(), "@inp     "); assert_eq!(app.input.history(app.input.history_len() - 1), Some("@binary")); assert!(app.pending.is_none());
        app.line_input("/paste".into()).unwrap(); app.line_input("@binary".into()).unwrap(); app.line_input("/end".into()).unwrap(); assert_eq!(app.multiline.as_deref(), Some("@binary")); assert_eq!(app.input.text(), "@inp     "); assert!(app.pending.is_none()); app.multiline = None; app.line_input("/paste".into()).unwrap(); app.line_input("/end".into()).unwrap(); assert!(app.multiline.is_none());
        for command in ["/hlep", "/help/", "/new/", "/model/"] { app.line_input(command.into()).unwrap(); assert!(app.pending.is_none()); }
        app.config.mode = super::super::args::Mode::Direct; app.direct_prefix_blocked = true; app.input.set("half-typed new draft".into()); app.queue("queued follow-up".into()).unwrap(); let queued = app.queued.pop().unwrap(); app.start(queued).unwrap(); assert_eq!(app.input.text(), "half-typed new draft"); assert_eq!(app.input.history(app.input.history_len() - 1), Some("queued follow-up"));
        for prompt in [vade_harness::agent::CONTINUATION_PROMPT, vade_harness::agent::TOOL_CONTINUATION_PROMPT] { let before = app.input.history_len(); app.start(prompt.into()).unwrap(); assert_eq!(app.input.text(), "half-typed new draft"); assert_eq!(app.input.history_len(), before); } app.config.mode = super::super::args::Mode::Demo;
        let sent = app.prepare_input("@input.txt".into()).unwrap(); app.start(sent).unwrap(); assert_eq!(app.input.history(app.input.history_len() - 1), Some("@input.txt"));
        app.input.finish_history();
        let history = std::fs::read(root.join(".vade/history")).unwrap(); assert!(!String::from_utf8_lossy(&history).contains("private file bytes")); assert!(!String::from_utf8_lossy(&history).contains("[attached files]"));
        assert!(String::from_utf8_lossy(&history).contains("@input.txt"), "attachment recall entry must reach disk before its privacy check");
        app.pending.take(); drop(app); drop(workspace); std::fs::remove_dir_all(root).unwrap(); std::fs::remove_file(outside).unwrap();
    }
}
