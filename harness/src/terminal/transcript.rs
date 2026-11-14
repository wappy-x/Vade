//! The bounded, width-aware transcript rows behind the terminal view.
use super::*;

#[derive(Clone, Default)]
pub(super) struct Transcript {
    pub(super) lines: VecDeque<TranscriptLine>,
    pub(super) rows: VecDeque<String>,
    pub(super) bytes: usize,
    pub(super) logical_lines: usize,
    pub(super) width: usize,
    pub(super) cells: usize,
    pub(super) protected: bool,
    pub(super) review_start: usize,
    pub(super) incomplete: bool,
    pub(super) row_limit: bool,
    pub(super) history_rows: usize,
    pub(super) last_line_rows: usize,
    pub(super) output_hidden: bool,
    pub(super) thinking_expanded: bool,
}
#[derive(Clone, Default)]
pub(super) struct TranscriptLine {
    pub(super) text: String,
    pub(super) thinking: Option<Thought>,
}
#[derive(Clone)]
pub(super) struct Thought {
    pub(super) tokens: Option<u64>,
    pub(super) elapsed: Option<Duration>,
    pub(super) start: String,
    pub(super) end: String,
    pub(super) limited: bool,
    pub(super) color: bool,
}
impl From<String> for TranscriptLine {
    fn from(text: String) -> Self {
        Self {
            text,
            thinking: None,
        }
    }
}
impl std::ops::Deref for TranscriptLine {
    type Target = String;
    fn deref(&self) -> &String {
        &self.text
    }
}
impl std::ops::DerefMut for TranscriptLine {
    fn deref_mut(&mut self) -> &mut String {
        &mut self.text
    }
}
impl Transcript {
    pub(super) fn push_thinking(
        &mut self,
        mut text: String,
        mut thought: Thought,
        width: usize,
    ) {
        // Thoughts share the transcript's byte and logical-line budgets. Keep a
        // visible summary even when one thought fills all retained history.
        let cap = MAX_TRANSCRIPT_BYTES - 128 * 1024 - 1;
        let mut end = text.len().min(cap);
        if let Some((at, _)) = text.match_indices('\n').nth(4096 - 512 - 2) {
            end = end.min(at);
        }
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        thought.limited |= end < text.len();
        text.truncate(end);
        if self.lines.back().is_some_and(|line| line.is_empty()) {
            self.lines.pop_back();
            self.logical_lines = self.logical_lines.saturating_sub(1);
        } else if !self.lines.is_empty() {
            self.bytes += 1;
        }
        self.bytes = self.bytes.saturating_add(text.len() + 1);
        self.logical_lines += 2 + text.bytes().filter(|b| *b == b'\n').count();
        self.lines.push_back(TranscriptLine {
            text,
            thinking: Some(thought),
        });
        self.lines.push_back(String::new().into());
        self.width = 0;
        self.push("", width);
    }
    pub(super) fn append_review(&mut self, text: &str, width: usize) {
        self.reflow(width);
        let start = self.lines.len().saturating_sub(1);
        if self.lines.is_empty() {
            self.lines.push_back(String::new().into());
            self.logical_lines += 1;
        }
        self.bytes += text.len();
        for c in text.chars() {
            if c == '\n' {
                self.lines.push_back(String::new().into());
                self.logical_lines += 1;
            } else {
                self.lines.back_mut().unwrap().push(c);
            }
        }
        // An incomplete view stays paused until a resize reflows it. Otherwise
        // only the unfinished last line and new lines need wrapping.
        if self.incomplete {
            return;
        }
        self.rows
            .truncate(self.rows.len().saturating_sub(self.last_line_rows));
        for index in start..self.lines.len() {
            let line = &self.lines[index];
            let rows = render::wrap_exact(&line.replace('\t', "    "), width);
            let too_wide = line.chars().any(|c| cell_width(c) > width);
            if !self.extend_review(rows, too_wide, width) {
                break;
            }
        }
        self.cells = self.rows.back().map_or(0, |r| render::width(r));
    }
    pub(super) fn extend_review(&mut self, rows: Vec<String>, too_wide: bool, width: usize) -> bool {
        while self.history_rows > 0 && self.rows.len() + rows.len() > MAX_APPROVAL_ROWS {
            self.rows.pop_front();
            self.history_rows -= 1;
        }
        self.row_limit = self.rows.len() + rows.len() > MAX_APPROVAL_ROWS;
        if self.row_limit || too_wide {
            self.incomplete = true;
            self.rows = render::wrap(APPROVAL_VIEW_ERROR, width).into();
            self.last_line_rows = 0;
            return false;
        }
        self.last_line_rows = rows.len();
        self.rows.extend(rows);
        true
    }
    pub(super) fn push(&mut self, text: &str, width: usize) {
        self.reflow(width);
        if self.lines.is_empty() {
            self.lines.push_back(String::new().into());
            self.logical_lines += 1;
        }
        for c in text.chars() {
            self.bytes += c.len_utf8();
            if c == '\n' {
                self.lines.push_back(String::new().into());
                self.logical_lines += 1;
            } else {
                self.lines.back_mut().unwrap().push(c);
            }
            self.wrap(c);
        }
        // Leave headroom so streaming a newline does not repeatedly reflow the
        // entire retained transcript once either limit is reached.
        let line_target = if self.logical_lines > 4096 {
            4096 - 512
        } else {
            4096
        };
        let byte_target = if self.bytes > MAX_TRANSCRIPT_BYTES {
            MAX_TRANSCRIPT_BYTES - 128 * 1024
        } else {
            MAX_TRANSCRIPT_BYTES
        };
        let mut trimmed = false;
        while self.lines.len() > 1 && (self.logical_lines > line_target || self.bytes > byte_target)
        {
            let line = self.lines.pop_front().unwrap();
            self.logical_lines = self
                .logical_lines
                .saturating_sub(1 + line.bytes().filter(|b| *b == b'\n').count());
            self.bytes = self.bytes.saturating_sub(line.len() + 1);
            trimmed = true;
        }
        if self.bytes > byte_target {
            let line = self.lines.front_mut().unwrap();
            let mut at = self.bytes - byte_target;
            while !line.is_char_boundary(at) {
                at += 1;
            }
            line.drain(..at);
            self.bytes -= at;
            trimmed = true;
        }
        if trimmed {
            self.width = 0;
            self.reflow(width);
        }
    }
    pub(super) fn push_styled(&mut self, text: &str, width: usize) {
        // Reuse bounded retention, then reflow the trusted SGR at cell width.
        self.push(text, width);
        self.width = 0;
        self.reflow(width);
    }
    pub(super) fn reflow(&mut self, width: usize) {
        if self.width == width {
            return;
        }
        self.width = width;
        self.rows.clear();
        self.incomplete = false;
        self.row_limit = false;
        self.history_rows = 0;
        self.last_line_rows = 0;
        for index in 0..self.lines.len() {
            let line = &self.lines[index];
            let rows = if let Some(thought) = &line.thinking {
                render::thinking(line, thought.tokens, thought.elapsed, (&thought.start, Some(&thought.end)), self.thinking_expanded, thought.limited, width, thought.color)
            } else if self.protected && index >= self.review_start {
                render::wrap_exact(&line.replace('\t', "    "), width)
            } else {
                render::wrap(&line.replace('\t', "    "), width)
            };
            if self.protected && index < self.review_start {
                // Only unrelated history may be discarded. A long old answer
                // must not prevent a small patch from being reviewed on resize.
                self.rows.extend(rows);
                while self.rows.len() > 16 * 1024 {
                    self.rows.pop_front();
                }
                self.history_rows = self.rows.len();
                continue;
            }
            if self.protected {
                let too_wide = line.chars().any(|c| cell_width(c) > width);
                if !self.extend_review(rows, too_wide, width) {
                    break;
                }
            } else {
                self.rows.extend(rows);
            }
        }
        while !self.protected && self.rows.len() > 16 * 1024 {
            self.rows.pop_front();
        }
        self.cells = self.rows.back().map_or(0, |r| render::width(r));
    }
    pub(super) fn wrap(&mut self, c: char) {
        if self.rows.is_empty() {
            self.rows.push_back(String::new());
        }
        if c == '\n' {
            self.rows.push_back(String::new());
            self.cells = 0;
        } else if c == '\t' {
            for _ in 0..4 {
                self.wrap(' ');
            }
        } else {
            let size = cell_width(c);
            if size > self.width {
                return;
            }
            if self.cells + size > self.width {
                self.rows.push_back(String::new());
                self.cells = 0;
            }
            self.rows.back_mut().unwrap().push(c);
            self.cells += size;
        }
        if self.rows.len() > 16 * 1024 {
            self.rows.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_reflows_unicode_and_bounds_logical_and_physical_history() {
        let mut t = Transcript::default();
        t.push("a你好\tb\nnext", 5);
        assert_eq!(
            t.rows.iter().cloned().collect::<Vec<_>>(),
            ["a你好", "    b", "next"]
        );
        t.reflow(3);
        assert!(
            t.rows
                .iter()
                .all(|r| r.chars().map(cell_width).sum::<usize>() <= 3)
        );
        // Word wrap may end a row at a space; every other character stays.
        assert_eq!(t.rows.iter().cloned().collect::<String>().replace(' ', ""), "a你好bnext");
        t.push(&"x".repeat(1024 * 1024 + 4), 1);
        assert!(t.bytes <= 1024 * 1024 && t.rows.len() <= 16 * 1024);
        t.push("\nend", 1);
        assert_eq!(t.rows.back().map(String::as_str), Some("d"));
        t.push(&"line\n".repeat(5000), 10);
        assert!(t.lines.len() <= 4096);
    }

    #[test]
    fn transcript_trims_in_batches_and_keeps_utf8_safe_headroom() {
        let mut t = Transcript::default();
        t.push(&"line\n".repeat(4096), 80);
        assert_eq!(t.lines.len(), 4096 - 512);
        let bytes = t.bytes;
        for _ in 0..400 {
            t.push("next\n", 80);
        }
        assert_eq!(t.lines.len(), 4096 - 512 + 400);
        assert_eq!(t.bytes, bytes + 400 * 5);
        assert_eq!(t.lines.front().unwrap().text, "line");
        assert_eq!(t.rows.len(), t.lines.len());
        let mut t = Transcript::default();
        t.push(&"🦀".repeat(1024 * 1024 / 4 + 1), 80);
        assert!(t.bytes <= 1024 * 1024 - 128 * 1024);
        let bytes = t.bytes;
        t.push(&"🦀".repeat(1024), 80);
        assert_eq!(t.bytes, bytes + 4096);
        assert!(t.lines.front().unwrap().starts_with('🦀'));
    }
}
