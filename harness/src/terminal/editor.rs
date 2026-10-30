//! UTF-8 scalar-boundary editor. Cursor state is independent of rendering.
use super::{Key, MAX_INPUT_BYTES, cell_width};
use crate::session::{MAX_INPUT_HISTORY_BYTES, MAX_INPUT_HISTORY_ENTRIES};
use std::collections::VecDeque;
use std::io::{self, Read, Seek, Write};
use crate::tools::Workspace;
use std::sync::Arc;

/// The chat input line: UTF-8 text, a cursor on scalar boundaries, and the
/// private, bounded recall history.
#[derive(Default, Debug)]
pub struct Editor {
    text: String,
    cursor: usize,
    typed: Vec<bool>,
    // Input recall is local UI state, independent of model context and replies.
    history: VecDeque<String>,
    history_bytes: usize,
    history_root: Option<Arc<Workspace>>,
    history_error: Option<String>,
    history_write: Option<(std::sync::mpsc::SyncSender<(Arc<Workspace>, String)>, std::sync::mpsc::Receiver<String>, std::thread::JoinHandle<()>)>,
    search: Option<(String, Option<usize>, String)>,
}
impl Editor {
    pub fn text(&self) -> &str {
        &self.text
    }
    pub fn cursor(&self) -> usize {
        self.cursor
    }
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
    pub fn len(&self) -> usize {
        self.text.len()
    }
    pub fn remember(&mut self, text: &str) {
        if text.trim().is_empty()
            || text.len() > MAX_INPUT_BYTES
            || matches!(text.trim(), "/approve" | "/deny" | "/approve session" | "/approve project" | "y" | "n" | "Y" | "N")
            || self.history.back().is_some_and(|previous| previous == text)
        {
            return;
        }
        while self.history.len() >= MAX_INPUT_HISTORY_ENTRIES
            || self.history_bytes.saturating_add(text.len()) > MAX_INPUT_HISTORY_BYTES
        {
            if let Some(previous) = self.history.pop_front() {
                self.history_bytes -= previous.len();
            } else {
                break;
            }
        }
        self.history.push_back(text.to_owned());
        self.history_bytes += text.len();
        if let Some(root) = &self.history_root {
            if let Err(error) = Self::append_history(root, text) {
                self.history_error = Some(format!("Input history not saved: {error}"));
            }
        }
    }
    /// Recall lives in the workspace's private state directory.
    pub fn load_history(&mut self, root: &Arc<Workspace>) -> io::Result<()> {
        self.finish_history();
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            let dir = root.state_dir()?;
            let _lock = dir.lock("history.lock")?;
            if let Some(mut file) = dir.optional_file("history")? {
                let compact = file.metadata()?.len() > (2 * MAX_INPUT_HISTORY_BYTES) as u64;
                self.history_root = None;
                loop {
                    let mut size = [0; 4];
                    if file.read(&mut size[..1])? == 0 {
                        break;
                    }
                    file.read_exact(&mut size[1..])?;
                    let size = u32::from_le_bytes(size) as usize;
                    if size > MAX_INPUT_BYTES {
                        return Err(io::Error::other("invalid input history frame"));
                    }
                    let mut bytes = vec![0; size];
                    file.read_exact(&mut bytes)?;
                    let text = String::from_utf8(bytes).map_err(io::Error::other)?;
                    self.remember(&text);
                }
                if compact {
                    let mut kept = Vec::with_capacity(self.history_bytes + 4 * self.history.len());
                    for text in &self.history {
                        kept.extend_from_slice(&(text.len() as u32).to_le_bytes());
                        kept.extend_from_slice(text.as_bytes());
                    }
                    dir.write_atomic("history", &kept)?;
                }
            }
            self.history_root = Some(Arc::clone(root));
            Ok(())
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = root;
            Err(io::Error::other(
                "private input history requires macOS or Linux",
            ))
        }
    }
    fn append_history(root: &Workspace, text: &str) -> io::Result<()> {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            let dir = root.state_dir()?;
            let _lock = dir.lock("history.lock")?;
            let mut file = dir.journal_file("history")?;
            let mut frame = (text.len() as u32).to_le_bytes().to_vec();
            frame.extend_from_slice(text.as_bytes());
            file.seek(io::SeekFrom::End(0))?;
            file.write_all(&frame)?;
            file.sync_data()
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = (root, text);
            Err(io::Error::other(
                "private input history requires macOS or Linux",
            ))
        }
    }
    pub fn history_error(&mut self) -> Option<String> {
        if let Some((_, errors, _)) = &self.history_write {
            for error in errors.try_iter() { self.history_error = Some(error); }
        }
        self.history_error.take()
    }
    /// Recall updates immediately; serialize disk writes away from UI input.
    pub fn remember_async(&mut self, text: &str) {
        let previous = self.history.back().cloned();
        let root = self.history_root.take();
        self.remember(text);
        self.history_root = root;
        if self.history.back() == previous.as_ref() {
            return;
        }
        if let Some(root) = self.history_root.clone() {
            if self.history_write.is_none() {
                let (tx, rx) = std::sync::mpsc::sync_channel::<(Arc<Workspace>, String)>(16);
                let (errors, error_rx) = std::sync::mpsc::channel();
                let worker = std::thread::spawn(move || {
                    for (root, text) in rx {
                        if let Err(error) = Self::append_history(&root, &text) {
                            let _ = errors.send(format!("Input history not saved: {error}"));
                        }
                    }
                });
                self.history_write = Some((tx, error_rx, worker));
            }
            if self.history_write.as_ref().unwrap().0.send((root, text.to_owned())).is_err() {
                self.history_error = Some("Input history not saved: history writer failed".into());
            }
        }
    }
    /// Finish pending recall persistence before shutdown or another history load.
    pub fn finish_history(&mut self) {
        if let Some((tx, errors, write)) = self.history_write.take() {
            drop(tx);
            if write.join().is_err() { self.history_error = Some("Input history not saved: history writer failed".into()); }
            for error in errors.try_iter() { self.history_error = Some(error); }
        }
    }
    pub fn search_prompt(&self) -> Option<String> {
        self.search.as_ref().map(|(query, _, _)| {
            format!("Ctrl+R history · {query:?} · Enter accepts · Esc restores")
        })
    }
    pub fn search_key(&mut self, key: &Key) -> bool {
        if self.search.is_none() && matches!(key, Key::Search) {
            self.search = Some((String::new(), None, self.text.clone()));
        }
        let Some((mut query, mut before, draft)) = self.search.take() else {
            return false;
        };
        match key {
            Key::Search => (),
            Key::Text(text) if query.len() + text.len() <= MAX_INPUT_BYTES => {
                query.push_str(text);
                before = None;
            }
            Key::Backspace => {
                query.pop();
                before = None;
            }
            Key::Enter => return true,
            Key::Escape | Key::Interrupt => {
                self.set(draft);
                return true;
            }
            _ => return false,
        }
        let end = before.unwrap_or(self.history.len());
        let found = (0..end).rev().find(|i| self.history[*i].contains(&query));
        if let Some(index) = found {
            self.set(self.history[index].clone());
            before = Some(index);
        }
        self.search = Some((query, before, draft));
        true
    }
    /// Return true only when the draft has an adjacent logical row to visit.
    pub fn move_row(&mut self, down: bool) -> bool {
        let adjacent = if down {
            self.text[self.cursor..].contains('\n')
        } else {
            self.text[..self.cursor].contains('\n')
        };
        if adjacent {
            self.vertical(down);
        }
        adjacent
    }
    pub fn continue_line(&mut self) -> bool {
        if self.text[..self.cursor].ends_with('\\')
            && (self.cursor == self.text.len() || self.text[self.cursor..].starts_with('\n'))
        {
            self.backspace();
            self.insert("\n");
            return true;
        }
        false
    }
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
    pub fn history(&self, index: usize) -> Option<&str> {
        self.history.get(index).map(String::as_str)
    }
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.history_bytes = 0;
    }
    pub fn set(&mut self, text: String) {
        self.cursor = text.len();
        self.typed = vec![false; text.len()];
        self.text = text;
    }
    pub fn set_typed(&mut self, text: String) {
        self.set(text);
        self.typed.fill(true);
    }
    /// The final-line approval shortcut requires freshly typed command bytes.
    /// Pasted/recalled multiline text cannot impersonate that shortcut.
    /// A typed one-line draft that is only an answer key (y, n, s, p) answers
    /// as its command; a pasted, recalled, or multiline key stays a draft.
    pub fn approval_command(&self) -> Option<(&str, usize)> {
        if !self.text.contains('\n') && self.typed.iter().all(|typed| *typed) {
            let command = match self.text.trim() {
                "y" | "Y" | "yes" | "Yes" => Some("/approve"),
                "n" | "N" | "no" | "No" => Some("/deny"),
                "s" | "S" => Some("/approve session"),
                "p" | "P" => Some("/approve project"),
                _ => None,
            };
            if let Some(command) = command {
                return Some((command, 0));
            }
        }
        let start = self.text.rfind('\n').map_or(0, |at| at + 1);
        let line = &self.text[start..];
        let command = line.trim();
        let offset = start + line.len() - line.trim_start().len();
        if matches!(command, "/approve" | "/deny" | "/approve session" | "/approve project")
            && self.typed[offset..offset + command.len()]
                .iter()
                .all(|typed| *typed)
        {
            Some((command, start.saturating_sub(1)))
        } else {
            None
        }
    }
    pub fn remove_approval_command(&mut self) {
        if let Some((_, start)) = self.approval_command() {
            self.text.truncate(start);
            self.typed.truncate(start);
            self.cursor = self.cursor.min(start);
        }
    }
    pub fn clear(&mut self) {
        self.text.clear();
        self.typed.clear();
        self.cursor = 0;
    }
    pub fn take(&mut self) -> String {
        self.cursor = 0;
        self.typed.clear();
        std::mem::take(&mut self.text)
    }
    pub fn insert(&mut self, text: &str) -> bool {
        self.insert_origin(text, true)
    }
    pub fn insert_paste(&mut self, text: &str) -> bool {
        self.insert_origin(text, false)
    }
    fn insert_origin(&mut self, text: &str, typed: bool) -> bool {
        if self.text.len().saturating_add(text.len()) > MAX_INPUT_BYTES {
            return false;
        }
        self.text.insert_str(self.cursor, text);
        self.typed.splice(
            self.cursor..self.cursor,
            std::iter::repeat_n(typed, text.len()),
        );
        self.cursor += text.len();
        true
    }
    pub fn left(&mut self) {
        if let Some((index, _)) = self.text[..self.cursor].char_indices().next_back() {
            self.cursor = index;
        }
    }
    pub fn right(&mut self) {
        if let Some(c) = self.text[self.cursor..].chars().next() {
            self.cursor += c.len_utf8();
        }
    }
    pub fn home(&mut self) {
        self.cursor = self.text[..self.cursor].rfind('\n').map_or(0, |i| i + 1);
    }
    pub fn end(&mut self) {
        self.cursor += self.text[self.cursor..]
            .find('\n')
            .unwrap_or(self.text.len() - self.cursor);
    }
    pub fn backspace(&mut self) {
        let end = self.cursor;
        self.left();
        self.text.drain(self.cursor..end);
        self.typed.drain(self.cursor..end);
    }
    pub fn delete(&mut self) {
        if let Some(c) = self.text[self.cursor..].chars().next() {
            self.text.drain(self.cursor..self.cursor + c.len_utf8());
            self.typed.drain(self.cursor..self.cursor + c.len_utf8());
        }
    }
    pub fn word_left(&mut self) {
        while self.text[..self.cursor]
            .chars()
            .next_back()
            .is_some_and(|c| !c.is_alphanumeric())
        {
            self.left();
        }
        while self.text[..self.cursor]
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric)
        {
            self.left();
        }
    }
    pub fn word_right(&mut self) {
        while self.text[self.cursor..]
            .chars()
            .next()
            .is_some_and(|c| !c.is_alphanumeric())
        {
            self.right();
        }
        while self.text[self.cursor..]
            .chars()
            .next()
            .is_some_and(char::is_alphanumeric)
        {
            self.right();
        }
    }
    pub fn delete_word_forward(&mut self) {
        let start = self.cursor;
        self.word_right();
        self.text.drain(start..self.cursor);
        self.typed.drain(start..self.cursor);
        self.cursor = start;
    }
    /// Delete to the end of the current line; on an empty line, join the next.
    pub fn kill_to_end(&mut self) {
        let end = self.text[self.cursor..]
            .find('\n')
            .map_or(self.text.len(), |n| self.cursor + n);
        if end == self.cursor && end < self.text.len() {
            self.text.remove(self.cursor);
            self.typed.remove(self.cursor);
        } else {
            self.text.drain(self.cursor..end);
            self.typed.drain(self.cursor..end);
        }
    }
    pub fn delete_word(&mut self) {
        let end = self.cursor;
        while self.text[..self.cursor]
            .chars()
            .next_back()
            .is_some_and(char::is_whitespace)
        {
            self.left();
        }
        while self.text[..self.cursor]
            .chars()
            .next_back()
            .is_some_and(|c| !c.is_whitespace())
        {
            self.left();
        }
        self.text.drain(self.cursor..end);
        self.typed.drain(self.cursor..end);
    }
    pub fn vertical(&mut self, down: bool) {
        let start = self.text[..self.cursor].rfind('\n').map_or(0, |i| i + 1);
        let column: usize = self.text[start..self.cursor].chars().map(cell_width).sum();
        let range = if down {
            let Some(end) = self.text[self.cursor..].find('\n').map(|n| n + self.cursor) else {
                return;
            };
            let next = end + 1;
            let end = self.text[next..]
                .find('\n')
                .map_or(self.text.len(), |n| next + n);
            next..end
        } else {
            if start == 0 {
                return;
            }
            let end = start - 1;
            let previous = self.text[..end].rfind('\n').map_or(0, |n| n + 1);
            previous..end
        };
        self.cursor = range.start;
        let mut used = 0;
        for c in self.text[range].chars() {
            let width = cell_width(c);
            if used + width > column {
                break;
            }
            used += width;
            self.cursor += c.len_utf8();
        }
    }
}

impl Drop for Editor {
    fn drop(&mut self) {
        self.finish_history();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unicode_cursor_inserts_deletes_and_preserves_exact_prompt() {
        let mut editor = Editor::default();
        editor.insert("a🦀éz");
        editor.left();
        editor.left();
        editor.insert("中");
        assert_eq!(editor.text(), "a🦀中éz");
        editor.backspace();
        editor.delete();
        assert_eq!(editor.text(), "a🦀z");
        editor.home();
        editor.delete();
        editor.end();
        assert_eq!(editor.take(), "🦀z");
        assert_eq!(editor.cursor(), 0);
    }
    #[test]
    fn multiline_moves_by_display_column_and_word_deletion_is_local() {
        let mut editor = Editor::default();
        editor.insert("abcd\n🦀 yz\nlast");
        assert!(editor.move_row(false));
        assert_eq!(editor.cursor(), "abcd\n🦀 y".len());
        assert!(editor.move_row(false));
        assert!(!editor.move_row(false));
        editor.set("abcd\n🦀 yz\nlast".into());
        editor.home();
        editor.right();
        editor.right();
        editor.vertical(false);
        assert_eq!(editor.cursor(), "abcd\n🦀".len());
        editor.end();
        editor.delete_word();
        assert_eq!(editor.text(), "abcd\n🦀 \nlast");
        editor.vertical(true);
        assert!(editor.text().is_char_boundary(editor.cursor()));
        editor.set("next\\".into());
        assert!(editor.continue_line());
        assert_eq!(editor.text(), "next\n");
    }
    #[test]
    fn word_motion_and_kill_to_end_respect_unicode_and_lines() {
        let mut editor = Editor::default();
        editor.insert("héllo wörld, src/main.rs\nsecond");
        while editor.cursor() > 0 {
            editor.left();
        }
        editor.word_right();
        assert_eq!(&editor.text()[..editor.cursor()], "héllo");
        editor.word_right();
        editor.word_right();
        assert_eq!(&editor.text()[..editor.cursor()], "héllo wörld, src");
        editor.word_left();
        assert_eq!(&editor.text()[..editor.cursor()], "héllo wörld, ");
        editor.kill_to_end();
        assert_eq!(editor.text(), "héllo wörld, \nsecond");
        editor.kill_to_end();
        assert_eq!(editor.text(), "héllo wörld, second");
        editor.home();
        editor.delete_word_forward();
        assert_eq!(editor.text(), " wörld, second");
    }
    #[test]
    fn oversized_paste_is_refused_without_partial_insertion() {
        let mut editor = Editor::default();
        editor.insert("keep");
        assert!(!editor.insert(&"x".repeat(MAX_INPUT_BYTES)));
        assert_eq!(editor.text(), "keep");
    }
    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[rustfmt::skip]
    fn oversized_history_compacts_on_load_and_remains_private() {
        use std::os::unix::fs::PermissionsExt;
        let root = crate::test_workspace::Fixture::new(); let mut editor = Editor::default();
        let workspace = Arc::new(root.workspace()); editor.load_history(&workspace).unwrap();
        for n in 0..1100 { editor.remember(&format!("{n:04}\n{}\n", "🦀".repeat(256))); }
        let path = root.0.join(".vade/history");
        assert!(path.metadata().unwrap().len() > (2 * MAX_INPUT_HISTORY_BYTES) as u64);
        let mut loaded = Editor::default(); loaded.load_history(&workspace).unwrap();
        assert_eq!(loaded.history, editor.history);
        assert!(path.metadata().unwrap().len() <= (MAX_INPUT_HISTORY_BYTES + 4 * MAX_INPUT_HISTORY_ENTRIES) as u64);
        assert_eq!(path.metadata().unwrap().permissions().mode() & 0o777, 0o600);
        for text in ["first", "first", "y", "second", "third"] { loaded.remember_async(text); }
        let expected = loaded.history.clone(); drop(loaded);
        let mut loaded = Editor::default(); loaded.load_history(&workspace).unwrap(); assert_eq!(loaded.history, expected);
        assert_eq!(loaded.history.iter().rev().take(3).map(String::as_str).collect::<Vec<_>>(), ["third", "second", "first"]);
    }
}

#[cfg(test)]
mod approval_tests {
    use super::Editor;

    #[test]
    fn only_typed_text_can_answer_an_approval_at_any_scope() {
        for command in ["/approve", "/deny", "/approve session", "/approve project"] {
            let mut typed = Editor::default();
            typed.insert(command);
            assert_eq!(typed.approval_command().map(|(c, _)| c), Some(command));
            let mut pasted = Editor::default();
            pasted.insert_paste(&format!("{command}\n"));
            assert_eq!(pasted.approval_command(), None, "{command}");
        }
    }

    #[test]
    fn only_a_typed_one_line_key_answers_an_approval() {
        for (key, command) in [("y", "/approve"), ("n", "/deny"), ("s", "/approve session"), ("p", "/approve project")] {
            let mut typed = Editor::default();
            typed.insert(key);
            assert_eq!(typed.approval_command(), Some((command, 0)));
            typed.remove_approval_command();
            assert!(typed.is_empty());
            let mut pasted = Editor::default();
            pasted.insert_paste(key);
            assert_eq!(pasted.approval_command(), None, "{key}");
            let mut pasted_line = Editor::default();
            pasted_line.insert_paste(&format!("{key}\n"));
            assert_eq!(pasted_line.approval_command(), None, "{key}");
            let mut recalled = Editor::default();
            recalled.set(key.into());
            assert_eq!(recalled.approval_command(), None, "{key}");
            let mut second_line = Editor::default();
            second_line.insert(&format!("note\n{key}"));
            assert_eq!(second_line.approval_command(), None, "{key}");
        }
    }
}
