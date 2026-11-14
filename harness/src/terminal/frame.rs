//! Terminal frame construction, dock layout and screen output.
use super::*;

impl Terminal {
    pub(super) fn draw_frame(
        &mut self,
        text: &str,
        cursor: usize,
        prefix: &str,
        status: &str,
        transient: &str,
        panel: &str,
        popup: &str,
    ) -> io::Result<()> {
        let (width, height) = (self.width(), self.height());
        if !self.fullscreen && !self.review_fullscreen && self.needs_review(width, height) {
            let frame = self.inline_bytes(&[], 0, 1);
            self.control(&frame)?;
            if self.interactive {
                self.control("\x1b[?1049h\x1b[2J\x1b[H")?;
            }
            self.review_fullscreen = true;
            self.screen_size = (0, 0);
        }
        let (lines, cursor_row, cursor_column) = self.frame_lines(
            text, cursor, prefix, status, transient, panel, popup, width, height,
        );
        if !self.fullscreen && !self.review_fullscreen {
            let column = cursor_column.min(width).max(1);
            if self.committed.is_empty()
                && self.screen_size == (width, height)
                && self.screen == lines
                && self.live_cursor_row == cursor_row
                && self.live_cursor_column == column
            {
                return Ok(());
            }
            let frame = self.inline_bytes(&lines, cursor_row, column);
            self.screen = lines;
            self.screen_size = (width, height);
            return self.control(&frame);
        }
        let frame = self.fullscreen_bytes(lines, cursor_row, cursor_column, width, height);
        self.control(&frame)
    }
    pub(super) fn fullscreen_bytes(&mut self, lines: Vec<String>, cursor_row: usize, cursor_column: usize, width: usize, height: usize) -> String {
        let changed_size = self.screen_size != (width, height);
        let mut frame = String::from("\x1b[?2026h\x1b[?25l");
        if changed_size {
            frame.push_str("\x1b[2J");
        }
        for (row, line) in lines.iter().enumerate() {
            if changed_size || self.screen.get(row) != Some(line) {
                frame.push_str(&format!("\x1b[{};1H\r\x1b[2K", row + 1));
                frame.push_str(line);
                if self.color {
                    frame.push_str("\x1b[0m");
                }
                if row + 1 < height {
                    frame.push_str("\x1b[1B\r");
                }
            }
        }
        frame.push_str(&format!(
            "\x1b[{};{}H\x1b[?25h\x1b[?2026l",
            cursor_row + 1,
            cursor_column.min(width).max(1)
        ));
        self.screen = lines;
        self.screen_size = (width, height);
        frame
    }
    /// No I/O: consume complete logical lines once, then replace only the dock.
    pub(super) fn inline_bytes(&mut self, lines: &[String], row: usize, column: usize) -> String {
        let end = self.committed.rfind('\n').map_or(0, |n| n + 1);
        let frame = live_frame(&self.committed[..end], lines, row, column, self.live_cursor_row);
        self.committed.drain(..end);
        self.live_cursor_row = row;
        self.live_cursor_column = column;
        frame
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn frame_lines(
        &mut self,
        text: &str,
        cursor: usize,
        prefix: &str,
        status: &str,
        transient: &str,
        panel: &str,
        popup: &str,
        width: usize,
        height: usize,
    ) -> (Vec<String>, usize, usize) {
        let cells = width.saturating_sub(1);
        let inline = self.raw && !self.fullscreen && !self.review_fullscreen
            && !self.needs_review(width, height);
        if !inline {
            self.transcript.reflow(cells.max(1));
        }
        if let Some(view) = &mut self.approval_view {
            view.reflow(cells.max(1));
        }
        let transcript = self.approval_view.as_ref().unwrap_or(&self.transcript);
        let status = if transcript.incomplete || (self.approval_view.is_some() && height < 3) {
            "Approval paused · enlarge terminal"
        } else {
            status
        };
        let input = render::editor(
            text,
            cursor,
            prefix,
            cells.max(1),
            height.saturating_sub(3).clamp(1, 8),
            !self.hint_seen,
            self.color,
        );
        let dock = input.lines.len() + usize::from(height > input.lines.len());
        let panel_budget = height.saturating_sub(dock + 2).min(17);
        let popup_rows: Vec<_> = popup.lines().take(panel_budget.min(10)).collect();
        let panel_rows: Vec<_> = panel
            .lines()
            .take(panel_budget.saturating_sub(popup_rows.len()))
            .collect();
        let room = height.saturating_sub(dock + panel_rows.len() + popup_rows.len());
        let transcript_height = if inline { room.min(height / 3) } else { room };
        let preview = if self.approval_view.is_some() {
            String::new()
        } else {
            if answer_text(&self.answer_preview).is_empty() {
                String::new()
            } else {
                let answer = answer_text(&self.answer_preview);
                let styled = render::markdown(answer, self.color);
                let tail = if inline { live_tail(&styled, cells, transcript_height) } else { &styled };
                format!("\n{tail}")
            }
        };
        let waiting = preview.is_empty() && !transient.is_empty() && transcript_height >= 4;
        let mut extra = if !self.thinking_preview.is_empty() && self.approval_view.is_none() {
            // One blank row above a live thought, as above a finished one.
            std::iter::once(String::new()).chain(render::thinking(
                if inline { live_tail(&self.thinking_preview, cells, transcript_height) } else { &self.thinking_preview },
                None,
                None,
                (&self.thinking_time, None),
                self.transcript.thinking_expanded,
                self.thinking_preview_limited,
                cells.max(1),
                self.color,
            )).collect()
        } else {
            Vec::new()
        };
        let reply_rows: Vec<String> = if !preview.is_empty() {
            render::wrap(&preview, cells.max(1))
        } else if waiting {
            // Transient waiting text occupies the answer's future position.
            // It never enters transcript history or survives the first delta.
            vec![render::style(&head(transient, cells), "2", self.color)]
        } else {
            Vec::new()
        };
        extra.extend(reply_rows);
        if let Some(pager) = &mut self.pager {
            let room = height.saturating_sub(1);
            let total = transcript.rows.len() + extra.len();
            pager.total = total;
            let index = (transcript.bytes, cells, transcript.thinking_expanded);
            if pager.indexed != Some(index) {
                pager.headers.clear();
                pager.hits.clear();
                for (row, text) in transcript.rows.iter().enumerate() {
                    let plain = render::plain(text);
                    if plain.starts_with(&format!("{} ", render::USER_BAR)) && !transcript.rows.get(row.wrapping_sub(1)).is_some_and(|above| render::plain(above).starts_with(render::USER_BAR)) {
                        pager.headers.push(row);
                    }
                    if !pager.query.is_empty() {
                        pager
                            .hits
                            .extend(plain.match_indices(&pager.query).map(|_| row));
                    }
                }
                pager.indexed = Some(index);
            }
            pager.hits.retain(|row| *row < transcript.rows.len());
            if !pager.query.is_empty() {
                for (row, text) in extra.iter().enumerate() {
                    pager.hits.extend(
                        render::plain(text)
                            .match_indices(&pager.query)
                            .map(|_| transcript.rows.len() + row),
                    );
                }
            }
            self.scroll = self.scroll.min(total.saturating_sub(room.max(1)));
            let end = total.saturating_sub(self.scroll);
            let start = end.saturating_sub(room);
            let turn = pager
                .headers
                .iter()
                .filter(|r| **r <= start)
                .count()
                .max(1)
                .min(pager.headers.len());
            let label = if pager.searching {
                format!("REVIEW · /{} · Enter search · Esc back", pager.query)
            } else if !pager.query.is_empty() {
                format!(
                    "REVIEW · /{} · {} matches · n/N next/prev · q back",
                    pager.query,
                    pager.hits.len()
                )
            } else {
                format!(
                    "REVIEW · turn {turn}/{} · / search · n next · {{ }} prompts · Ctrl+T thinking · q back",
                    pager.headers.len()
                )
            };
            let mut lines = vec![render::clip(&label, cells)];
            lines.extend(
                window(&transcript.rows, &extra, start, end)
                    .into_iter()
                    .map(|row| {
                        let plain = render::plain(&row);
                        if pager.query.is_empty() || !plain.contains(&pager.query) {
                            row
                        } else {
                            plain.replace(&pager.query, &format!("\x1b[7m{}\x1b[27m", pager.query))
                        }
                    }),
            );
            lines.resize(height, String::new());
            return (lines, 0, 1);
        }
        let history = if inline { &VecDeque::new() } else { &transcript.rows };
        let total = history.len() + extra.len();
        if !inline {
            self.scroll = self.scroll.min(total.saturating_sub(transcript_height.max(1)));
        }
        let end = total.saturating_sub(if inline { 0 } else { self.scroll });
        let start = end.saturating_sub(transcript_height);
        let mut lines = window(history, &extra, start, end);
        if !inline {
            lines.resize(transcript_height, String::new());
        }
        lines.extend(panel_rows.iter().map(|line| head(line, cells)));
        if height > input.lines.len() {
            // A short screen has no answer position: fold the spinner into
            // the footer instead of dropping it.
            let joined = |a: &str, b: &str| -> String {
                if a.is_empty() {
                    b.to_owned()
                } else if b.is_empty() {
                    a.to_owned()
                } else {
                    format!("{a} · {b}")
                }
            };
            let status = if !waiting && !transient.is_empty() && preview.is_empty() {
                joined(transient, status)
            } else {
                status.to_owned()
            };
            let status = if self.scroll > 0 {
                let hint = "history · PgDn";
                let room = cells.saturating_sub(render::width(hint) + 3);
                joined(&fit_status(&status, room), hint)
            } else {
                fit_status(&status, cells)
            };
            lines.push(render::budget_footer(&head(&status, cells), self.color));
        }
        // The menu touches the input's top edge; no status row splits them.
        lines.extend(popup_rows.iter().map(|line| render::clip(line, cells)));
        let input_row = lines.len();
        let cursor_row = input_row + input.cursor_row;
        let cursor_column = input.cursor_column + 1;
        lines.extend(input.lines);
        (lines, cursor_row, cursor_column)
    }
    pub(super) fn control(&mut self, bytes: &str) -> io::Result<()> {
        let mut out = io::stdout().lock();
        // Check under the same stdout lock used by panic restoration: an
        // already-built frame must never clear a diagnostic after restoration.
        if self.raw && !self.raw() {
            return Ok(());
        }
        out.write_all(bytes.as_bytes())?;
        out.flush()
    }
}

/// Relative moves keep committed paragraphs soft-wrapped in native scrollback.
/// IND reserves dock rows, scrolling only when they would pass the screen edge.
fn live_frame(
    committed: &str,
    lines: &[String],
    row: usize,
    column: usize,
    previous_row: usize,
) -> String {
    let mut out = String::from("\x1b[?2026h\x1b[?25l\r");
    if previous_row > 0 {
        out.push_str(&format!("\x1b[{previous_row}A"));
    }
    out.push_str("\x1b[J");
    out.push_str(&committed.replace('\n', "\r\n"));
    let last = lines.len().saturating_sub(1);
    if last > 0 {
        out.push_str(&"\x1bD".repeat(last));
        out.push_str(&format!("\x1b[{last}A\r"));
    }
    for (index, line) in lines.iter().enumerate() {
        if index > 0 {
            out.push_str("\x1b[1B\r");
        }
        out.push_str(line);
        out.push_str("\x1b[0m");
    }
    if last > row {
        out.push_str(&format!("\x1b[{}A", last - row));
    }
    out.push('\r');
    if column > 1 {
        out.push_str(&format!("\x1b[{}C", column - 1));
    }
    out.push_str("\x1b[?25h\x1b[?2026l");
    out
}

pub(super) fn live_tail(text: &str, width: usize, rows: usize) -> &str {
    if rows == 0 {
        return "";
    }
    let mut start = text.len().saturating_sub((width * rows * 4).clamp(1, 4096));
    while !text.is_char_boundary(start) {
        start += 1;
    }
    start = text[..start].rfind('\n').map_or(start, |at| at + 1);
    let tail = &text[start..];
    tail.rmatch_indices('\n').nth(rows - 1).map_or(tail, |(at, _)| &tail[at + 1..])
}

/// Rows `start..end` of the retained rows followed by `extra`, cloning only
/// the visible window rather than the whole transcript each frame.
pub(super) fn window(rows: &VecDeque<String>, extra: &[String], start: usize, end: usize) -> Vec<String> {
    let split = rows.len();
    let mut lines: Vec<String> = rows
        .range(start.min(split)..end.min(split))
        .cloned()
        .collect();
    if end > split {
        let from = start.saturating_sub(split);
        lines.extend(extra[from..end - split].iter().cloned());
    }
    lines
}

/// Keep the leading item and the trailing hint of a ` · ` separated footer;
/// drop middle facts from the right until the row fits.
pub(super) fn fit_status(status: &str, width: usize) -> String {
    if render::width(status) <= width {
        return status.to_owned();
    }
    if let Some(rest) = status.strip_prefix("╰ ") {
        // Run footer: clock · elapsed · tokens · write speed · less useful
        // facts. Drop from the right, then the clock; a cut-off hint stays.
        let mut parts: Vec<_> = rest.split(" · ").collect();
        let pinned = parts.last().filter(|p| p.starts_with("cut off")).copied();
        parts.retain(|part| Some(*part) != pinned);
        let fit = |parts: &[&str]| {
            let line = format!("╰ {}", parts.iter().copied().chain(pinned).collect::<Vec<_>>().join(" · "));
            (render::width(&line) <= width).then_some(line)
        };
        while parts.len() > 4 {
            parts.pop();
            if let Some(line) = fit(&parts) { return line; }
        }
        if parts.len() > 1 {
            parts.remove(0);
        }
        return fit(&parts).unwrap_or_else(|| head(&format!("╰ {}", parts.join(" · ")), width));
    }
    let parts: Vec<&str> = status.split(" · ").collect();
    if parts.len() <= 2 {
        return head(status, width);
    }
    let last = parts[parts.len() - 1];
    let mut kept = parts[..parts.len() - 1].to_vec();
    while kept.len() > 1 {
        kept.pop();
        let candidate = format!("{} · {last}", kept.join(" · "));
        if render::width(&candidate) <= width {
            return candidate;
        }
    }
    head(&format!("{} · {last}", kept[0]), width)
}

fn head(text: &str, width: usize) -> String {
    let text = sanitize(text).replace('\t', "    ").replace('\n', " ");
    let mut cells = 0;
    let mut result = String::new();
    for c in text.chars() {
        let size = cell_width(c);
        if cells + size > width {
            break;
        }
        result.push(c);
        cells += size;
    }
    result
}
