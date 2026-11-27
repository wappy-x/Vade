//! Bounded incremental decoding of server-sent events.
use super::*;

#[derive(Default)]
pub(crate) struct SseDecoder {
    pub(super) line: Vec<u8>,
    data: String,
    pub(super) has_data: bool,
    skip_lf: bool,
    first_line: bool,
    started: bool,
    pub(super) done: bool,
}

impl SseDecoder {
    pub(crate) fn at_event_boundary(&self) -> bool {
        self.line.is_empty() && !self.has_data
    }

    pub(crate) fn feed(
        &mut self,
        bytes: &[u8],
        event: &mut dyn FnMut(&str) -> io::Result<bool>,
    ) -> io::Result<()> {
        let mut rest = bytes;
        while !rest.is_empty() && !self.done {
            if self.skip_lf {
                self.skip_lf = false;
                if rest[0] == b'\n' {
                    rest = &rest[1..];
                    continue;
                }
            }
            // Common single-data-line events borrow the socket buffer directly.
            // Fragmented/multiline events use the bounded reusable scratch below.
            if self.started && self.line.is_empty() && !self.has_data && rest.starts_with(b"data:")
            {
                if let Some(end) = rest.iter().position(|&b| b == b'\n' || b == b'\r') {
                    let delimiter = if rest[end..].starts_with(b"\n\n") {
                        2
                    } else if rest[end..].starts_with(b"\r\n\r\n") {
                        4
                    } else {
                        0
                    };
                    if delimiter != 0 {
                        if end > MAX_EVENT {
                            return Err(invalid("SSE line exceeds limit"));
                        }
                        let line = std::str::from_utf8(&rest[5..end])
                            .map_err(|_| invalid("SSE is not UTF-8"))?;
                        let data = line.strip_prefix(' ').unwrap_or(line);
                        self.done = event(data)?;
                        rest = &rest[end + delimiter..];
                        continue;
                    }
                }
            }
            let take = rest
                .iter()
                .position(|&b| b == b'\n' || b == b'\r')
                .unwrap_or(rest.len());
            if self.line.len().saturating_add(take) > MAX_EVENT {
                return Err(invalid("SSE line exceeds limit"));
            }
            self.line.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if let Some(&byte) = rest.first() {
                self.skip_lf = byte == b'\r';
                self.dispatch_line(event)?;
                rest = &rest[1..];
            }
        }
        Ok(())
    }

    fn dispatch_line(&mut self, event: &mut dyn FnMut(&str) -> io::Result<bool>) -> io::Result<()> {
        let mut line = std::str::from_utf8(&self.line).map_err(|_| invalid("SSE is not UTF-8"))?;
        if !self.started {
            self.first_line = true;
            self.started = true;
        }
        if self.first_line {
            line = line.strip_prefix('\u{feff}').unwrap_or(line);
            self.first_line = false;
        }
        if line.is_empty() {
            if self.has_data {
                self.data.pop(); // SSE joins data fields with one newline.
                self.done = event(&self.data)?;
                self.data.clear();
                self.has_data = false;
            }
        } else if !line.starts_with(':') {
            let (name, value) = line.split_once(':').map_or((line, ""), |(name, value)| {
                (name, value.strip_prefix(' ').unwrap_or(value))
            });
            if name == "data" {
                if self.data.len().saturating_add(value.len() + 1) > MAX_EVENT {
                    return Err(invalid("SSE event exceeds limit"));
                }
                self.data.push_str(value);
                self.data.push('\n');
                self.has_data = true;
            }
        }
        self.line.clear();
        Ok(())
    }
}
