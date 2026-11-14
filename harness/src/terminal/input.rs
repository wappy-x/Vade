//! Terminal input channels and incremental key decoding.
use super::*;

/// One decoded key press or pasted text from the terminal.
#[derive(Debug, PartialEq, Eq)]
pub enum Key {
    Text(String),
    Enter,
    Newline,
    Backspace,
    Clear,
    DeleteWord,
    Up,
    Down,
    Left,
    Right,
    Home,
    LineEnd,
    Delete,
    Escape,
    Interrupt,
    End,
    PasteStart,
    PasteEnd,
    Tab,
    PageUp,
    PageDown,
    WheelUp,
    WheelDown,
    WordLeft,
    WordRight,
    LineUp,
    LineDown,
    DeleteWordForward,
    KillToEnd,
    Redraw,
    CtrlT,
    CtrlO,
    Search,
}
/// What the standard-input reader thread delivers: raw bytes in a raw
/// terminal, whole lines otherwise, or end of input and its failures.
#[derive(Debug)]
pub enum Input {
    Bytes(Vec<u8>),
    Line(String),
    Eof,
    TooLong,
    InvalidUtf8,
    Error,
}

/// Start the thread that reads standard input, in raw byte mode or line mode,
/// and return its channel.
pub fn input_channel(raw: bool) -> Receiver<Input> {
    let (tx, rx) = wake::channel(16);
    std::thread::spawn(move || {
        let mut stdin = io::stdin().lock();
        let mut buffer = [0u8; 1024];
        let mut line = Vec::new();
        let mut overflow = false;
        loop {
            match stdin.read(&mut buffer) {
                Ok(0) => {
                    if !raw && (!line.is_empty() || overflow) {
                        send_line(&tx, &mut line, overflow);
                    }
                    let _ = tx.send(Input::Eof);
                    break;
                }
                Ok(count) if raw => {
                    if tx.send(Input::Bytes(buffer[..count].to_vec())).is_err() {
                        break;
                    }
                }
                Ok(count) => {
                    for byte in &buffer[..count] {
                        if *byte == b'\n' {
                            if !send_line(&tx, &mut line, overflow) {
                                return;
                            }
                            overflow = false;
                        } else if line.len() < MAX_INPUT_BYTES {
                            line.push(*byte);
                        } else {
                            overflow = true;
                        }
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    std::thread::park_timeout(Duration::from_millis(20));
                }
                Err(_) => {
                    let _ = tx.send(Input::Error);
                    break;
                }
            }
        }
    });
    rx
}
fn send_line(tx: &Sender<Input>, line: &mut Vec<u8>, overflow: bool) -> bool {
    let bytes = std::mem::take(line);
    if overflow {
        return tx.send(Input::TooLong).is_ok();
    }
    let mut value = match String::from_utf8(bytes) {
        Ok(value) => value,
        Err(_) => return tx.send(Input::InvalidUtf8).is_ok(),
    };
    if value.ends_with('\r') {
        value.pop();
    }
    tx.send(Input::Line(value)).is_ok()
}

/// Turns raw terminal bytes into [`Key`]s: escape sequences, bracketed paste,
/// Meta chords and split UTF-8.
#[derive(Default)]
pub struct Decoder {
    escape: Vec<u8>,
    escaped_at: Option<Instant>,
    escape_expired: bool,
    // A Meta chord arrives as ESC plus its letter in one read; an ESC in an
    // earlier read followed by a letter is Esc and then that letter.
    feed_serial: u64,
    escape_feed: u64,
    utf8: Vec<u8>,
    paste: bool,
    paste_cr: bool,
}
impl Decoder {
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<Key> {
        let mut keys = Vec::new();
        self.feed_serial = self.feed_serial.wrapping_add(1);
        for &byte in bytes {
            if !self.escape.is_empty() {
                self.escape.push(byte);
                let mut retype = false;
                if self.escape == b"\x1b[200~" {
                    self.paste = true;
                    self.paste_cr = false;
                    keys.push(Key::PasteStart);
                    self.clear_escape();
                } else if self.escape == b"\x1b[201~" {
                    self.paste = false;
                    self.paste_cr = false;
                    keys.push(Key::PasteEnd);
                    self.clear_escape();
                } else if !self.paste && matches!(self.escape.as_slice(), b"\x1b[A" | b"\x1bOA") {
                    keys.push(Key::Up);
                    self.clear_escape();
                } else if !self.paste && matches!(self.escape.as_slice(), b"\x1b[B" | b"\x1bOB") {
                    keys.push(Key::Down);
                    self.clear_escape();
                } else if !self.paste
                    && self.escape.starts_with(b"\x1b[<")
                    && matches!(byte, b'M' | b'm')
                {
                    if byte == b'M' {
                        if let Ok(report) =
                            std::str::from_utf8(&self.escape[3..self.escape.len() - 1])
                        {
                            let fields: Vec<_> = report.split(';').map(str::parse::<u16>).collect();
                            if let [Ok(button), Ok(_), Ok(_)] = fields.as_slice() {
                                match button & 0xc3 {
                                    64 => keys.push(Key::WheelUp),
                                    65 => keys.push(Key::WheelDown),
                                    _ => (),
                                }
                            }
                        }
                    }
                    self.clear_escape();
                } else if !self.paste
                    && matches!(
                        self.escape.as_slice(),
                        b"\x1b[C"
                            | b"\x1b[D"
                            | b"\x1b[H"
                            | b"\x1b[F"
                            | b"\x1b[3~"
                            | b"\x1b[5~"
                            | b"\x1b[6~"
                            | b"\x1bOC"
                            | b"\x1bOD"
                            | b"\x1bOH"
                            | b"\x1bOF"
                            | b"\x1b[1;5C"
                            | b"\x1b[1;5D"
                            | b"\x1b[1;3C"
                            | b"\x1b[1;3D"
                            | b"\x1b[1;5A"
                            | b"\x1b[1;5B"
                            | b"\x1b[1;3A"
                            | b"\x1b[1;3B"
                    )
                {
                    keys.push(match self.escape.as_slice() {
                        b"\x1b[C" | b"\x1bOC" => Key::Right,
                        b"\x1b[D" | b"\x1bOD" => Key::Left,
                        b"\x1b[H" | b"\x1bOH" => Key::Home,
                        b"\x1b[F" | b"\x1bOF" => Key::LineEnd,
                        b"\x1b[5~" => Key::PageUp,
                        b"\x1b[6~" => Key::PageDown,
                        b"\x1b[1;5C" | b"\x1b[1;3C" => Key::WordRight,
                        b"\x1b[1;5D" | b"\x1b[1;3D" => Key::WordLeft,
                        b"\x1b[1;5A" | b"\x1b[1;3A" => Key::LineUp,
                        b"\x1b[1;5B" | b"\x1b[1;3B" => Key::LineDown,
                        _ => Key::Delete,
                    });
                    self.clear_escape();
                } else if !self.paste
                    && self.escape.starts_with(b"\x1b[")
                    && matches!(byte, b'u' | b'~')
                {
                    if let Some(key) = modified_key(&self.escape) {
                        keys.push(key);
                    }
                    self.clear_escape();
                } else if !self.paste
                    && !self.escape_expired
                    && (self.escape == b"\x1b\r" || self.escape == b"\x1b\n")
                {
                    keys.push(Key::Newline);
                    self.clear_escape();
                } else if self.escape.len() == 2 && byte != b'[' && byte != b'O' {
                    // Meta chords (Option-as-Meta on macOS, Alt elsewhere).
                    // Unknown chords are dropped whole, never typed as text.
                    let same_read = self.escape_feed == self.feed_serial;
                    let expired = self.escape_expired;
                    self.clear_escape();
                    if !self.paste {
                        if expired {
                            retype = true;
                        } else if same_read {
                            keys.push(match byte {
                                b'b' => Key::WordLeft,
                                b'f' => Key::WordRight,
                                b'd' => Key::DeleteWordForward,
                                8 | 127 => Key::DeleteWord,
                                _ => Key::Escape,
                            });
                        } else {
                            keys.push(Key::Escape);
                            retype = true;
                        }
                    }
                } else if self.escape.len() > 2 && (0x40..=0x7e).contains(&byte)
                    || self.escape.len() > 64
                {
                    self.clear_escape();
                }
                if !retype {
                    continue;
                }
            }
            if byte == 27 {
                self.flush_utf8(&mut keys);
                self.escape.push(byte);
                self.escaped_at = Some(Instant::now());
                self.escape_feed = self.feed_serial;
                continue;
            }
            if self.paste {
                match byte {
                    b'\r' => {
                        self.flush_utf8(&mut keys);
                        keys.push(Key::Newline);
                    }
                    b'\n' if self.paste_cr => (),
                    b'\n' => {
                        self.flush_utf8(&mut keys);
                        keys.push(Key::Newline);
                    }
                    b'\t' => {
                        self.flush_utf8(&mut keys);
                        keys.push(Key::Text("\t".into()));
                    }
                    0..=31 | 127 => (),
                    _ => self.character(byte, &mut keys),
                }
                self.paste_cr = byte == b'\r';
                continue;
            }
            let key = match byte {
                b'\r' => Some(Key::Enter),
                b'\n' => Some(Key::Newline),
                3 => Some(Key::Interrupt),
                4 => Some(Key::End),
                8 | 127 => Some(Key::Backspace),
                21 => Some(Key::Clear),
                23 => Some(Key::DeleteWord),
                11 => Some(Key::KillToEnd),
                12 => Some(Key::Redraw),
                20 => Some(Key::CtrlT),
                15 => Some(Key::CtrlO),
                18 => Some(Key::Search),
                1 => Some(Key::Home),
                5 => Some(Key::LineEnd),
                b'\t' => Some(Key::Tab),
                0..=31 => None,
                _ => {
                    self.character(byte, &mut keys);
                    continue;
                }
            };
            self.flush_utf8(&mut keys);
            if let Some(key) = key {
                keys.push(key);
            }
        }
        keys
    }
    pub fn timeout(&mut self) -> Option<Key> {
        // A paste terminator may cross reads or pause while the terminal sends
        // a large clipboard. Never expire it into ordinary, sendable input.
        if !self.paste
            && self.escape.len() == 1
            && self
                .escaped_at
                .is_some_and(|at| at.elapsed() > Duration::from_millis(50))
        {
            // Keep the prefix after delivering a lone Esc. A delayed CSI,
            // particularly a paste opener, still belongs to this sequence.
            self.escape_expired = true;
            self.escaped_at = None;
            return Some(Key::Escape);
        }
        None
    }
    fn clear_escape(&mut self) {
        self.escape.clear();
        self.escaped_at = None;
        self.escape_expired = false;
    }
    fn character(&mut self, byte: u8, keys: &mut Vec<Key>) {
        self.utf8.push(byte);
        match std::str::from_utf8(&self.utf8) {
            Ok(value) => {
                let value = sanitize(value);
                if let Some(Key::Text(previous)) = keys.last_mut() {
                    previous.push_str(&value);
                } else {
                    keys.push(Key::Text(value));
                }
                self.utf8.clear();
            }
            Err(error) if error.error_len().is_some() || self.utf8.len() >= 4 => {
                self.flush_utf8(keys)
            }
            _ => (),
        }
    }
    fn flush_utf8(&mut self, keys: &mut Vec<Key>) {
        if !self.utf8.is_empty() {
            keys.push(Key::Text(sanitize(&String::from_utf8_lossy(&self.utf8))));
            self.utf8.clear();
        }
    }
}

/// CSI u (kitty/fixterms) and xterm modifyOtherKeys share the modifier mask:
/// one plus Shift=1, Alt=2, Ctrl=4. Release reports never become keystrokes.
fn modified_key(sequence: &[u8]) -> Option<Key> {
    let text = std::str::from_utf8(sequence.strip_prefix(b"\x1b[")?).ok()?;
    let (code, modifiers, shifted) = if let Some(text) = text.strip_suffix('u') {
        let mut fields = text.split(';');
        let mut codes = fields.next()?.split(':');
        let code = codes.next()?.parse::<u32>().ok()?;
        let shifted = codes
            .next()
            .filter(|value| !value.is_empty())
            .and_then(|value| value.parse().ok());
        let mut modifier = fields.next().unwrap_or("1").split(':');
        let modifiers = modifier.next()?.parse::<u32>().ok()?.checked_sub(1)?;
        let event = modifier.next().unwrap_or("1").parse::<u8>().ok()?;
        if !matches!(event, 1 | 2) {
            return None;
        }
        (code, modifiers, shifted)
    } else {
        let mut fields = text.strip_suffix('~')?.split(';');
        if fields.next()? != "27" {
            return None;
        }
        let modifiers = fields.next()?.parse::<u32>().ok()?.checked_sub(1)?;
        let code = fields.next()?.parse::<u32>().ok()?;
        if fields.next().is_some() {
            return None;
        }
        (code, modifiers, None)
    };
    // Caps/Num Lock do not change shortcuts; Super/Hyper/Meta belong to the
    // terminal or desktop, including the macOS clipboard shortcut Cmd+C.
    if modifiers & !0xc7 != 0 {
        return None;
    }
    let modifiers = modifiers & 7;
    match code {
        13 | 57414 => {
            return Some(if modifiers == 0 {
                Key::Enter
            } else {
                Key::Newline
            });
        }
        10 => return Some(Key::Newline),
        9 => return Some(Key::Tab),
        27 => return Some(Key::Escape),
        8 | 127 => {
            return Some(if modifiers & 6 != 0 {
                Key::DeleteWord
            } else {
                Key::Backspace
            });
        }
        _ => (),
    }
    if modifiers & 4 != 0 {
        return match code {
            97 | 65 => Some(Key::Home),
            99 | 67 => Some(Key::Interrupt),
            100 | 68 => Some(Key::End),
            101 | 69 => Some(Key::LineEnd),
            105 | 73 => Some(Key::Tab),
            106 | 74 => Some(Key::Newline),
            107 | 75 => Some(Key::KillToEnd),
            108 | 76 => Some(Key::Redraw),
            109 | 77 => Some(Key::Enter),
            116 | 84 => Some(Key::CtrlT),
            111 | 79 => Some(Key::CtrlO),
            114 | 82 => Some(Key::Search),
            117 | 85 => Some(Key::Clear),
            119 | 87 => Some(Key::DeleteWord),
            _ => None,
        };
    }
    if modifiers & 2 != 0 {
        return match code {
            98 | 66 => Some(Key::WordLeft),
            102 | 70 => Some(Key::WordRight),
            100 | 68 => Some(Key::DeleteWordForward),
            _ => None,
        };
    }
    let c = char::from_u32(if modifiers & 1 != 0 {
        shifted.unwrap_or(code)
    } else {
        code
    })?;
    if c.is_control() || ('\u{e000}'..='\u{f8ff}').contains(&c) {
        return None;
    }
    let text = if modifiers & 1 != 0 && shifted.is_none() {
        c.to_uppercase().collect()
    } else {
        c.to_string()
    };
    Some(Key::Text(sanitize(&text)))
}

/// True once the terminal's interrupt signal (Ctrl+C) has arrived.
pub fn interrupted() -> bool {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        unix::INTERRUPTED.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragmented_unicode_and_navigation() {
        let mut d = Decoder::default();
        assert!(d.feed(&[0xe2, 0x82]).is_empty());
        assert_eq!(d.feed(&[0xac]), [Key::Text("€".into())]);
        assert!(d.feed(b"\x1b[").is_empty());
        assert_eq!(d.feed(b"A"), [Key::Up]);
        assert_eq!(
            d.feed(b"\r\x03\x7f"),
            [Key::Enter, Key::Interrupt, Key::Backspace]
        );
    }

    #[test]
    fn fragmented_wheel_reports_cannot_recall_prompts_or_enter_pasted_text() {
        let mut d = Decoder::default();
        assert!(d.feed(b"\x1b[<64;12;").is_empty());
        assert_eq!(d.feed(b"5M\x1b[<81;12;5M"), [Key::WheelUp, Key::WheelDown]);
        assert!(d.feed(b"\x1b[<0;12;5M\x1b[<0;12;5m").is_empty());
        assert_eq!(d.feed(b"\x1b[A\x1b[B"), [Key::Up, Key::Down]);
        assert_eq!(
            d.feed(b"\x1b[200~\x1b[<64;12;5M\x1b[201~"),
            [Key::PasteStart, Key::PasteEnd]
        );
    }

    #[test]
    fn bracketed_paste_does_not_submit_commands() {
        let mut d = Decoder::default();
        let keys = d.feed(b"\x1b[200~/quit\n\x03\x1b[201~");
        assert_eq!(keys.first(), Some(&Key::PasteStart));
        assert_eq!(keys.last(), Some(&Key::PasteEnd));
        assert!(!keys.contains(&Key::Enter));
        assert!(!keys.contains(&Key::Interrupt));
        assert!(keys.contains(&Key::Newline));
    }

    #[test]
    fn paste_normalizes_cr_lf_without_losing_blank_lines() {
        let mut decoder = Decoder::default();
        let keys = decoder.feed(b"\x1b[200~\r\nfirst\r\rsecond\n\x1b[201~");
        assert_eq!(keys.iter().filter(|key| **key == Key::Newline).count(), 4);
        assert!(!keys.contains(&Key::Enter));
        assert_eq!(
            decoder.feed(b"\x1b[13;2u\x1b[27;2;13~\x1b\r\n\x14"),
            [
                Key::Newline,
                Key::Newline,
                Key::Newline,
                Key::Newline,
                Key::CtrlT
            ]
        );
        assert!(decoder.feed(b"\x1b[13;2:3u\x1b[99;9u").is_empty());
        assert_eq!(
            decoder.feed(b"\x1b[99;5u\x1b[27;5;106~"),
            [Key::Interrupt, Key::Newline]
        );
    }

    #[test]
    fn meta_chords_and_modified_arrows_move_by_word_instead_of_clearing() {
        let mut d = Decoder::default();
        assert_eq!(
            d.feed(b"\x1bb\x1bf\x1bd\x1b\x7f\x1b[1;5C\x1b[1;5D\x1b[1;3C\x1b[1;3D\x0b\x0c\x1bOC"),
            [
                Key::WordLeft,
                Key::WordRight,
                Key::DeleteWordForward,
                Key::DeleteWord,
                Key::WordRight,
                Key::WordLeft,
                Key::WordRight,
                Key::WordLeft,
                Key::KillToEnd,
                Key::Redraw,
                Key::Right,
            ]
        );
        // An unknown chord is still a lone Escape; its letter never types.
        assert_eq!(d.feed(b"\x1bq"), [Key::Escape]);
        // Esc in one read and a letter in the next is Esc, then that letter.
        assert!(d.feed(b"\x1b").is_empty());
        assert_eq!(d.feed(b"b"), [Key::Escape, Key::Text("b".into())]);
        assert!(d.feed(b"\x1b").is_empty());
        assert_eq!(d.feed(b"\x7f"), [Key::Escape, Key::Backspace]);
        assert!(d.feed(b"\x1b").is_empty());
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(d.timeout(), Some(Key::Escape));
    }

    #[test]
    fn navigation_and_large_paste_decode_without_per_character_work_events() {
        let mut decoder = Decoder::default();
        assert_eq!(
            decoder.feed(b"\x1b[C\x1b[D\x1b[H\x1b[F\x1b[3~"),
            [Key::Right, Key::Left, Key::Home, Key::LineEnd, Key::Delete]
        );
        let mut bytes = b"\x1b[200~".to_vec();
        bytes.extend(vec![b'x'; MAX_INPUT_BYTES]);
        bytes.extend_from_slice(b"\x1b[201~");
        let keys = decoder.feed(&bytes);
        assert_eq!(keys.len(), 3);
        assert!(matches!(&keys[1], Key::Text(text) if text.len() == MAX_INPUT_BYTES));
        assert!(!keys.contains(&Key::Enter));
    }
}
