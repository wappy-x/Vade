//! A bounded terminal transcript and fixed input dock, with a line-mode fallback.
use crate::wake::{self, Sender};
use std::collections::VecDeque;
use std::io::{self, IsTerminal, Read, Write};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

pub mod clock;
pub mod clipboard;
pub mod editor;
pub mod render;
mod frame;
mod input;
mod transcript;
mod text;
pub use text::{answer_text, escape_preview, preview_char_needs_escape, reply_tail, sanitize};
pub(crate) use text::AnswerPrefix;
use text::cell_width;
use transcript::{Thought, Transcript};
use frame::fit_status;
pub use input::{Decoder, Input, Key, input_channel, interrupted};

pub const MAX_INPUT_BYTES: usize = 64 * 1024;
const MAX_APPROVAL_BYTES: usize = 256 * 1024;
const MAX_APPROVAL_ROWS: usize = 128 * 1024;
const MAX_REVIEW_HISTORY_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_TRANSCRIPT_BYTES: usize = 1024 * 1024;
const REVIEW_OUTPUT_HIDDEN: &str = "\n[later output hidden until approval closes]\n";
const APPROVAL_VIEW_ERROR: &str = "Approval unavailable: the terminal cannot show the full preview at this size. Enlarge the terminal to view it in full; after a refusal, request the operation again. If it still does not fit, request smaller patches.";

/// The chat display: a bounded transcript above a fixed input dock in a raw
/// terminal, or plain line output when the terminal is not interactive.
pub struct Terminal {
    raw: bool,
    fullscreen: bool,
    review_fullscreen: bool,
    committed: String,
    live_cursor_row: usize,
    live_cursor_column: usize,
    color: bool,
    interactive: bool,
    transcript: Transcript,
    approval_view: Option<Transcript>,
    answer_preview: String,
    answer_time: String,
    date: String,
    thinking_preview: String,
    thinking_preview_limited: bool,
    thinking_started: Option<Instant>,
    thinking_time: String,
    hint_seen: bool,
    screen: Vec<String>,
    screen_size: (usize, usize),
    scroll: usize,
    pager: Option<Pager>,
    plain_state: Option<String>,
    plain_stream_started: bool,
    plain_line_open: bool,
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    _guard: Option<unix::Guard>,
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    _exit_signals: Option<unix::Signals>,
}
impl Terminal {
    pub fn open(force_plain: bool) -> Self {
        let interactive = io::stdin().is_terminal() && io::stdout().is_terminal();
        let fullscreen = std::env::var("VADE_TUI").is_ok_and(|v| v == "fullscreen");
        let capable = std::env::var("TERM").map_or(true, |v| v != "dumb");
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        let guard = if interactive && capable && !force_plain {
            unix::Guard::enter(fullscreen).ok()
        } else {
            None
        };
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        let raw = guard.is_some();
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        let raw = false;
        let color = raw && std::env::var_os("NO_COLOR").is_none();
        let mut terminal = Self {
            raw,
            fullscreen,
            review_fullscreen: false,
            committed: String::new(),
            live_cursor_row: 0,
            live_cursor_column: 1,
            color,
            interactive,
            transcript: Transcript::default(),
            approval_view: None,
            answer_preview: String::new(),
            answer_time: String::new(),
            date: String::new(),
            thinking_preview: String::new(),
            thinking_preview_limited: false,
            thinking_started: None,
            thinking_time: String::new(),
            hint_seen: false,
            screen: Vec::new(),
            screen_size: (0, 0),
            scroll: 0,
            pager: None,
            plain_state: None,
            plain_stream_started: false,
            plain_line_open: false,
            #[cfg(any(target_os = "macos", target_os = "linux"))]
            _guard: guard,
            #[cfg(any(target_os = "macos", target_os = "linux"))]
            _exit_signals: None,
        };
        terminal.date_rule();
        terminal
    }
    /// A delegated child must be canceled and reaped even in line/one-shot mode.
    /// Raw mode already owns the same handlers through its terminal guard.
    pub fn watch_exit_signals(&mut self) {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        if !self.raw() && self._exit_signals.is_none() {
            self._exit_signals = Some(unix::Signals::install());
        }
    }
    pub fn raw(&self) -> bool {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        if self._guard.as_ref().is_some_and(|guard| !guard.active()) {
            return false;
        }
        self.raw
    }
    /// Restore the normal screen before leaving a compact, durable receipt.
    /// Plain/redirected output keeps its existing answer-only stdout contract.
    pub fn finish(&mut self, summary: &str) -> io::Result<()> {
        self.close_pager()?;
        if self.raw() && !self.fullscreen {
            self.close_approval();
            if !self.committed.is_empty() && !self.committed.ends_with('\n') {
                self.committed.push('\n');
            }
            let frame = self.inline_bytes(&[], 0, 1);
            self.control(&frame)?;
        }
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        if let Some(guard) = self._guard.take() {
            // Restore the screen/termios now, but retain blocking stdout until
            // the receipt is flushed. Dropping first reinstates O_NONBLOCK.
            guard.restore();
            self.raw = false;
            self.line(summary)?;
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        let _ = summary;
        Ok(())
    }
    pub fn interactive(&self) -> bool {
        self.interactive
    }
    pub fn notify_run(&mut self, elapsed: Duration, message: &str) -> io::Result<()> {
        if self.interactive && elapsed >= Duration::from_secs(30) {
            let message = sanitize(message).replace(['\r', '\n'], " ");
            self.control(&format!("\x07\x1b]9;{message}\x07\x1b]2;{message}\x07"))?;
        }
        Ok(())
    }
    pub fn width(&self) -> usize {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            unix::width().unwrap_or(80).clamp(1, 500)
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            80
        }
    }
    pub fn height(&self) -> usize {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            unix::dimensions()
                .map_or(24, |(_, height)| height)
                .clamp(1, 500)
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            24
        }
    }
    pub fn text(&mut self, text: &str) -> io::Result<()> {
        let text = sanitize(text);
        if self.raw() {
            self.retain(&text, false);
            Ok(())
        } else {
            let mut out = io::stdout().lock();
            out.write_all(text.as_bytes())?;
            if !text.is_empty() {
                self.plain_line_open = !text.ends_with('\n');
            }
            out.flush()
        }
    }
    fn retain(&mut self, text: &str, styled: bool) {
        self.retain_at(text, styled, false);
    }
    /// `exact` commits cell-exact rows (approval previews): a word moved to
    /// the next row could read as a diff line of its own.
    fn retain_at(&mut self, text: &str, styled: bool, exact: bool) {
        let width = self.width().saturating_sub(1).max(1);
        if !self.fullscreen {
            // Pre-wrapped at word boundaries: native soft wrap splits words.
            let text = text.replace('\t', "    ");
            let rows = if exact { render::wrap_exact(&text, width) } else { render::wrap(&text, width) };
            self.committed.push_str(&rows.join("\n"));
        }
        if styled {
            self.transcript.push_styled(text, width);
        } else {
            self.transcript.push(text, width);
        }
        if let Some(view) = &mut self.approval_view {
            // Keep status/help output visible without ever trimming the edit.
            // Excess unrelated output stays in ordinary bounded history.
            if !view.output_hidden
                && view
                    .bytes
                    .saturating_add(text.len() + REVIEW_OUTPUT_HIDDEN.len())
                    <= MAX_REVIEW_HISTORY_BYTES
            {
                view.append_review(text, width);
            } else if !view.output_hidden {
                view.append_review(REVIEW_OUTPUT_HIDDEN, width);
                view.output_hidden = true;
            }
        }
    }
    pub fn line(&mut self, text: &str) -> io::Result<()> {
        self.text(&format!("{text}\n"))
    }
    pub fn heading(&mut self, text: &str) -> io::Result<()> {
        self.styled_line(text, "1;36")
    }
    pub fn note(&mut self, text: &str) -> io::Result<()> {
        self.styled_line(text, "2")
    }
    /// A one-line outcome that needs attention (yellow when color is on).
    pub fn alert(&mut self, text: &str) -> io::Result<()> {
        self.flush_stream()?;
        self.styled_line(text, "33")
    }
    pub fn date_rule(&mut self) {
        if self.raw() {
            self.date = clock::now().1;
            let date = render::style(&self.date, "2", self.color);
            self.retain(&format!("\n{date}\n"), true);
        }
    }
    pub fn user_message(&mut self, text: &str) -> io::Result<()> {
        self.hint_seen = true;
        if self.raw() {
            if self.date != clock::now().1 {
                self.date_rule();
            }
            self.gap();
            self.retain(&format!("{}\n", render::user(text, self.color)), true);
            Ok(())
        } else {
            self.line(&format!("\n{}", render::user(text, false)))
        }
    }
    /// Exactly one blank line before a new block (You, Thinking, a tool call,
    /// Answer, Warning), however the previous output ended.
    fn gap(&mut self) {
        if !self.raw() {
            return;
        }
        let lines = &self.transcript.lines;
        let open = lines.back().is_some_and(|line| !line.is_empty());
        let blank = lines.len() >= 2
            && lines[lines.len() - 2].is_empty()
            && lines[lines.len() - 2].thinking.is_none();
        if open {
            self.retain("\n\n", true);
        } else if lines.len() > 1 && !blank {
            self.retain("\n", true);
        }
    }
    fn styled_line(&mut self, text: &str, sgr: &str) -> io::Result<()> {
        if self.raw() {
            let text = sanitize(text)
                .split('\n')
                .map(|line| render::style(line, sgr, self.color))
                .collect::<Vec<_>>()
                .join("\n");
            self.retain(&format!("{text}\n"), true);
            Ok(())
        } else {
            self.line(text)
        }
    }
    pub fn color(&self) -> bool {
        self.color
    }
    pub fn preview_answer(&mut self, text: &str) {
        if self.answer_time.is_empty() {
            self.answer_time = clock::now().0;
        }
        self.answer_preview.push_str(text);
    }
    pub fn discard_preview(&mut self) {
        self.answer_preview.clear();
        self.answer_time.clear();
    }
    pub fn end_round(&mut self) {
        self.discard_preview();
        self.plain_stream_started = false;
        self.plain_state = None;
    }
    pub fn preview_thinking(&mut self, text: &str, skipped: bool) {
        if self.thinking_time.is_empty() {
            self.thinking_time = clock::now().0;
        }
        self.thinking_started.get_or_insert_with(Instant::now);
        let mut end = text
            .len()
            .min(MAX_TRANSCRIPT_BYTES.saturating_sub(self.thinking_preview.len()));
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        self.thinking_preview.push_str(&text[..end]);
        self.thinking_preview_limited |= skipped || end < text.len();
    }
    pub fn discard_thinking_preview(&mut self) {
        self.thinking_started = None;
        self.thinking_time.clear();
        self.thinking_preview.clear();
        self.thinking_preview_limited = false;
    }
    /// A finished thought keeps its clocks, length and elapsed preview time.
    pub fn thinking(&mut self, text: &str, tokens: Option<u64>, limited: bool) -> io::Result<()> {
        self.thinking_preview.clear();
        self.thinking_preview_limited = false;
        let elapsed = self.thinking_started.take().map(|started| started.elapsed());
        let end = clock::now().0;
        let start = if self.thinking_time.is_empty() { end.clone() } else { std::mem::take(&mut self.thinking_time) };
        let thought = Thought { tokens, elapsed, start, end, limited, color: self.color };
        if self.raw() && !text.is_empty() {
            self.gap();
            if !self.fullscreen {
                let summary = render::thinking(
                    text, tokens, elapsed, (&thought.start, Some(&thought.end)), false, limited,
                    self.width().saturating_sub(1).max(1), self.color,
                );
                self.committed.push_str(&format!("{}\n", summary.join("\n")));
            }
            let width = self.width().saturating_sub(1).max(1);
            self.transcript
                .push_thinking(sanitize(text), thought, width);
        } else if self.interactive && !self.plain_stream_started && !text.is_empty() {
            for row in render::thinking(text, tokens, elapsed, (&thought.start, Some(&thought.end)), false, limited, usize::MAX, false) {
                self.line(&render::clip(&row.replace(" · Ctrl+T expand", ""), self.width()))?;
            }
        }
        Ok(())
    }
    pub fn toggle_thinking(&mut self) {
        self.transcript.thinking_expanded = !self.transcript.thinking_expanded;
        self.transcript.width = 0;
        if let Some(view) = &mut self.approval_view {
            view.thinking_expanded = self.transcript.thinking_expanded;
            view.width = 0;
        }
        self.scroll = 0;
    }
    pub fn answer(&mut self, text: &str) -> io::Result<()> {
        self.discard_preview();
        if self.raw() {
            let text = render::reply(answer_text(text), self.color);
            self.gap();
            self.retain(&format!("{text}\n"), true);
            Ok(())
        } else {
            self.line(answer_text(text))
        }
    }
    pub fn answer_status(&mut self, status: &str) -> io::Result<()> {
        self.flush_stream()?;
        // One row in the TUI: the least useful facts give way at narrow widths
        // (/status keeps them). Plain and redirected output keep every fact.
        if self.raw() {
            let width = self.width().saturating_sub(1).max(1);
            self.note(&fit_status(status, width))
        } else {
            self.note(status)
        }
    }
    /// A warning block in three plain parts; see `render::warning`.
    pub fn warning(&mut self, what: &str, doing: &str, next: &str) -> io::Result<()> {
        self.flush_stream()?;
        let text = render::warning(what, doing, next, self.color);
        if self.raw() {
            self.gap();
            self.retain(&format!("{text}\n"), true);
            Ok(())
        } else {
            self.line(&format!("\n{text}"))
        }
    }
    /// A tool call row: `● Read src/main.rs`.
    pub fn tool_call(&mut self, label: &str) -> io::Result<()> {
        if self.raw() {
            self.gap();
            let text = format!(
                "{} {}\n",
                render::style("●", "1;36", self.color),
                sanitize(label)
            );
            self.retain(&text, true);
            Ok(())
        } else {
            self.line(label)
        }
    }
    /// A tool result row under its call: `  ⎿ exit 0 · 12 lines · 0.3s`.
    pub fn tool_result(&mut self, summary: &str, failed: bool) -> io::Result<()> {
        if self.raw() {
            let text = format!(
                "  {} {}\n",
                render::style("⎿", "2", self.color),
                render::style(
                    &format!("{}{}", if failed { "✗ " } else { "" }, sanitize(summary)),
                    if failed { "31" } else { "2" },
                    self.color
                )
            );
            self.retain(&text, true);
            Ok(())
        } else {
            self.line(&format!("  {summary}"))
        }
    }
    /// A unified diff preview with colored +/- lines (raw mode only).
    pub fn diff(&mut self, text: &str) -> io::Result<()> {
        if self.raw() {
            let text = render::diff(&sanitize(text), self.color);
            self.retain(&format!("{text}\n"), true);
            Ok(())
        } else {
            self.line(text)
        }
    }
    fn approval(&mut self) -> io::Result<()> {
        if self.raw() {
            self.retain(&format!("{}\n", render::approval(self.color)), true);
            Ok(())
        } else {
            self.line(&render::approval(false))
        }
    }
    /// The approval lane owns a frozen, lossless view. Ordinary history may
    /// trim, but cannot evict any of this view while an answer is pending.
    /// Size-only failures retain a paused view; acceptance must check readiness.
    pub fn approval_preview(&mut self, title: &str, detail: &str) -> io::Result<()> {
        if title.len().saturating_add(detail.len()) > MAX_APPROVAL_BYTES {
            return Err(io::Error::other(
                "Approval refused: preview exceeds the terminal's 256 KiB review limit. Request smaller patches to view every change in full.",
            ));
        }
        if title.chars().any(preview_char_needs_escape)
            || detail.chars().any(preview_char_needs_escape)
        {
            return Err(io::Error::other(
                "Approval refused: preview contains characters the terminal would hide. Request an escaped, complete preview.",
            ));
        }
        let view = if self.raw() {
            let detail = if title == "Patch preview" {
                render::diff(detail, self.color)
            } else {
                detail.to_owned()
            };
            let text = format!(
                "{}\n{detail}\n{}\n",
                render::style(title, "2", self.color),
                render::approval(self.color)
            );
            let mut view = self.transcript.clone();
            view.protected = true;
            view.review_start = view.lines.len().saturating_sub(1);
            view.width = 0;
            view.append_review(&text, self.width().saturating_sub(1).max(1));
            if view.row_limit {
                return Err(io::Error::other(APPROVAL_VIEW_ERROR));
            }
            // The inline rows show the same exact text the protected view holds.
            self.retain_at(&text, true, true);
            Some(view)
        } else {
            self.note(title)?;
            if title == "Patch preview" {
                self.diff(detail)?;
            } else {
                self.line(detail)?;
            }
            self.approval()?;
            None
        };
        self.approval_view = view;
        if self.approval_view.is_some() {
            self.scroll = 0;
        }
        Ok(())
    }
    /// Recheck the current size at the acceptance boundary, even if a resize
    /// and an approval key arrive before the next frame is drawn.
    pub fn approval_ready(&mut self) -> io::Result<()> {
        let width = self.width().saturating_sub(1).max(1);
        self.approval_ready_at(width, self.height())
    }
    fn approval_ready_at(&mut self, width: usize, height: usize) -> io::Result<()> {
        if height < 3 {
            return Err(io::Error::other(APPROVAL_VIEW_ERROR));
        }
        if let Some(view) = &mut self.approval_view {
            view.reflow(width);
            if view.incomplete {
                return Err(io::Error::other(APPROVAL_VIEW_ERROR));
            }
        } else if self.raw() {
            return Err(io::Error::other(
                "Approval refused: complete preview is no longer available; request a new preview.",
            ));
        }
        Ok(())
    }
    pub fn close_approval(&mut self) {
        if self.approval_view.take().is_some() {
            self.scroll = 0;
        }
        if self.review_fullscreen && self.pager.is_none() {
            if self.interactive {
                let _ = self.control("\x1b[?1049l");
            }
            self.review_fullscreen = false;
            self.screen.clear();
            self.screen_size = (0, 0);
        }
    }
    fn needs_review(&mut self, width: usize, height: usize) -> bool {
        let Some(view) = &mut self.approval_view else {
            return false;
        };
        view.reflow(width.saturating_sub(1).max(1));
        view.incomplete || view.rows.len().saturating_sub(view.history_rows) > height.saturating_sub(4)
    }
    // Output is retained independently of the dock. Clearing a prompt must
    // never move the transcript cursor or put a transient status in history.
    pub fn clear_editor(&mut self) -> io::Result<()> {
        Ok(())
    }

    pub fn resized(&mut self) -> io::Result<()> {
        if self.raw() {
            self.screen.clear();
            self.screen_size = (0, 0);
        }
        Ok(())
    }
    pub fn scroll(&mut self, down: bool) {
        let page = self.height().saturating_sub(5).max(1);
        self.scroll_by(page, down);
    }
    pub fn wheel(&mut self, down: bool) {
        self.scroll_by(3, down);
    }
    fn scroll_by(&mut self, rows: usize, down: bool) {
        let transcript = self.approval_view.as_ref().unwrap_or(&self.transcript);
        self.scroll = if down {
            self.scroll.saturating_sub(rows)
        } else {
            self.scroll
                .saturating_add(rows)
                .min(
                    self.pager.as_ref().map_or(transcript.rows.len(), |p| p.total)
                        .saturating_sub(1),
                )
        };
    }
    pub fn pager_active(&self) -> bool {
        self.pager.is_some()
    }
    fn close_pager(&mut self) -> io::Result<()> {
        if let Some(pager) = self.pager.take() {
            self.control("\x1b[?1007l")?;
            if !self.fullscreen && (pager.entered_screen || self.approval_view.is_none()) {
                self.control("\x1b[?1049l")?;
                self.review_fullscreen = false;
            }
            self.scroll = pager.saved_scroll;
            self.resized()?;
        }
        Ok(())
    }
    /// Pager keys never reach the draft, history recall or approval handler.
    pub fn pager_key(&mut self, key: &Key) -> io::Result<bool> {
        if self.pager.is_none() {
            if *key != Key::CtrlO || !self.raw() {
                return Ok(false);
            }
            let entered_screen = !self.fullscreen && !self.review_fullscreen;
            if entered_screen {
                let frame = self.inline_bytes(&[], 0, 1);
                self.control(&frame)?;
                self.control("\x1b[?1049h\x1b[2J\x1b[H")?;
                self.review_fullscreen = true;
            }
            self.pager = Some(Pager {
                entered_screen,
                saved_scroll: self.scroll,
                ..Pager::default()
            });
            self.scroll = 0;
            self.control("\x1b[?1007h")?;
            self.resized()?;
            self.frame_lines("", 0, "", "", "", "", "", self.width(), self.height());
            return Ok(true);
        }
        if let Key::Text(text) = key {
            let pager = self.pager.as_ref().unwrap();
            if text.chars().count() > 1 && !pager.searching && !pager.pasting {
                for c in text.chars() {
                    if !self.pager_active() { break; }
                    self.pager_key(&Key::Text(c.to_string()))?;
                }
                return Ok(true);
            }
        }
        let page = self.height().saturating_sub(1).max(1);
        self.scroll = self
            .scroll
            .min(self.pager.as_ref().unwrap().total.saturating_sub(page));
        match key {
            Key::PasteStart => self.pager.as_mut().unwrap().pasting = true,
            Key::PasteEnd => self.pager.as_mut().unwrap().pasting = false,
            _ => (),
        }
        if self
            .pager
            .as_ref()
            .is_some_and(|p| p.pasting && !p.searching)
        {
            return Ok(true);
        }
        if self.pager.as_ref().is_some_and(|p| p.searching)
            && !matches!(key, Key::Escape | Key::Interrupt | Key::CtrlO)
        {
            let pager = self.pager.as_mut().unwrap();
            match key {
                Key::Text(text) if pager.query.len() + text.len() <= 4096 => {
                    pager.query.push_str(text)
                }
                Key::Backspace => {
                    pager.query.pop();
                }
                Key::Clear => pager.query.clear(),
                Key::Enter => pager.searching = false,
                _ => (),
            }
            pager.indexed = None;
            pager.hit = None;
            if *key == Key::Enter {
                self.pager_jump(true, false);
            }
            return Ok(true);
        }
        match key {
            Key::CtrlO | Key::Escape | Key::Interrupt => self.close_pager()?,
            Key::Text(text) if text == "q" => self.close_pager()?,
            Key::PageUp => self.scroll_by(page, false),
            Key::PageDown => self.scroll_by(page, true),
            Key::Text(text) if text == " " || text == "b" => self.scroll_by(page, text == " "),
            Key::Up | Key::WheelUp => self.scroll_by(1, false),
            Key::Down | Key::WheelDown => self.scroll_by(1, true),
            Key::Text(text) if text == "g" => self.scroll = usize::MAX,
            Key::Text(text) if text == "G" => self.scroll = 0,
            Key::Text(text) if text == "/" => {
                let pager = self.pager.as_mut().unwrap();
                pager.searching = true;
                pager.query.clear();
                pager.indexed = None;
                pager.hit = None;
            }
            Key::Text(text) if text == "n" || text == "N" => self.pager_jump(text == "n", false),
            Key::Text(text) if text == "{" || text == "}" => self.pager_jump(text == "}", true),
            Key::CtrlT => self.toggle_thinking(),
            Key::Redraw => self.resized()?,
            _ => (),
        }
        Ok(true)
    }
    fn pager_jump(&mut self, next: bool, prompt: bool) {
        // Refresh indexes before navigation, including keys from the same read.
        self.frame_lines("", 0, "", "", "", "", "", self.width(), self.height());
        let room = self.height().saturating_sub(1).max(1);
        let pager = self.pager.as_mut().unwrap();
        let start = pager.total.saturating_sub(self.scroll + room);
        let target = if prompt {
            if next {
                pager.headers.iter().copied().find(|r| *r > start)
            } else {
                pager.headers.iter().rev().copied().find(|r| *r < start)
            }
        } else if pager.hits.is_empty() {
            None
        } else {
            let count = pager.hits.len();
            let index = pager.hit.map_or_else(
                || {
                    if next {
                        pager.hits.iter().position(|r| *r >= start).unwrap_or(0)
                    } else {
                        count - 1
                    }
                },
                |i| {
                    if next {
                        (i + 1) % count
                    } else {
                        (i + count - 1) % count
                    }
                },
            );
            pager.hit = Some(index);
            Some(pager.hits[index])
        };
        if let Some(row) = target {
            self.scroll = pager.total.saturating_sub(row + room);
        }
    }
    pub fn editor(&mut self, text: &str, multiline: bool) -> io::Result<()> {
        self.editor_at(text, text.len(), multiline)
    }
    pub fn editor_at(&mut self, text: &str, cursor: usize, multiline: bool) -> io::Result<()> {
        self.editor_with_popup(text, cursor, multiline, "", "")
    }
    /// `status` is the idle footer (model, folder, session, context use).
    pub fn editor_with_popup(
        &mut self,
        text: &str,
        cursor: usize,
        multiline: bool,
        status: &str,
        popup: &str,
    ) -> io::Result<()> {
        if !self.raw() {
            if self.interactive {
                self.text(if multiline { "... " } else { "> " })?;
            }
            return Ok(());
        }
        self.draw_frame(
            text,
            cursor,
            if multiline { "+ " } else { "› " },
            if multiline {
                "multiline · /end sends · /cancel clears"
            } else {
                status
            },
            "",
            "",
            popup,
        )
    }
    /// Nothing of the answer is visible yet (hidden reasoning counts as nothing).
    pub fn preview_is_empty(&self) -> bool {
        answer_text(&self.answer_preview).is_empty()
    }

    /// Plain output marks only your turns (the bar); the answer needs no
    /// heading, so this is a no-op kept for the stream call sites.
    pub fn answer_heading(&mut self) -> io::Result<()> {
        Ok(())
    }
    pub fn stream_text(&mut self, text: &str) -> io::Result<()> {
        self.plain_stream_started |= !text.is_empty();
        self.text(text)
    }
    pub fn flush_stream(&mut self) -> io::Result<()> {
        if !self.raw() && self.interactive && self.plain_line_open {
            self.line("")?;
        }
        Ok(())
    }

    pub fn begin_run(&mut self) {
        self.answer_preview.clear();
        self.answer_time.clear();
        self.thinking_preview.clear();
        self.thinking_preview_limited = false;
        self.thinking_started = None;
        self.thinking_time.clear();
        self.plain_state = None;
        self.plain_stream_started = false;
    }

    /// A bounded metadata panel shares the redrawable frame with the prompt;
    /// it never enters the provisional model text or transcript scrollback.
    pub fn active_editor_with_panel(
        &mut self,
        text: &str,
        cursor: usize,
        state: &str,
        started: Instant,
        queued: usize,
        panel: &str,
    ) -> io::Result<()> {
        if !self.raw() {
            // Preserve contiguous plain reply deltas even when the worker
            // publishes Generating only after delivering its first delta.
            if self.interactive
                && !self.plain_stream_started
                && self.plain_state.as_deref() != Some(state)
            {
                self.note(&format!(
                    "{state} · {:.1}s",
                    started.elapsed().as_secs_f32()
                ))?;
                self.plain_state = Some(state.into());
            }
            return Ok(());
        }
        let status = format!(
            "{state} · {:.1}s{}",
            started.elapsed().as_secs_f32(),
            if queued > 0 {
                format!(" · {queued} queued")
            } else {
                String::new()
            }
        );
        self.draw_frame(text, cursor, "› ", &status, "", panel, "")
    }
    /// `status` fills the footer row; `transient` (spinner, state, clock) is
    /// drawn where the answer will appear while no answer text is visible.
    pub fn active_editor_with_popup(
        &mut self,
        text: &str,
        cursor: usize,
        status: &str,
        transient: &str,
        panel: &str,
        popup: &str,
    ) -> io::Result<()> {
        self.draw_frame(text, cursor, "› ", status, transient, panel, popup)
    }
}

#[derive(Default)]
struct Pager {
    entered_screen: bool,
    saved_scroll: usize,
    searching: bool,
    pasting: bool,
    query: String,
    hit: Option<usize>,
    hits: Vec<usize>,
    headers: Vec<usize>,
    total: usize,
    indexed: Option<(usize, usize, bool)>,
}


#[cfg(any(target_os = "macos", target_os = "linux"))]
mod unix;

#[cfg(test)]
mod tests;
