//! Exact answer bytes, independent of terminal rendering.
use std::io::{self, Write};
use std::process::{Command, Stdio};

pub const MAX_COPY_BYTES: usize = 64 * 1024;

fn within_cap(text: &str) -> io::Result<()> {
    if text.len() > MAX_COPY_BYTES {
        return Err(io::Error::other(
            "Copy refused: text exceeds the 64 KiB clipboard limit.",
        ));
    }
    Ok(())
}

/// Return the interior of the last closed Markdown fence without changing bytes.
pub fn last_code(text: &str) -> Option<&str> {
    let mut open = None;
    let mut last = None;
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let row = line.trim_end_matches(['\r', '\n']);
        let fence = row.trim_start_matches(' ');
        if row.len() - fence.len() <= 3 {
            let mark = fence.as_bytes().first().copied().unwrap_or(0);
            let count = fence.bytes().take_while(|b| *b == mark).count();
            if matches!(mark, b'`' | b'~') && count >= 3 {
                match open {
                    Some((start, expected, width))
                        if mark == expected
                            && count >= width
                            && fence[count..].trim().is_empty() =>
                    {
                        last = Some(&text[start..offset]);
                        open = None;
                    }
                    None if mark != b'`' || !fence[count..].contains('`') => {
                        open = Some((offset + line.len(), mark, count));
                    }
                    _ => (),
                }
            }
        }
        offset += line.len();
    }
    last
}

fn base64(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let a = chunk[0] as usize;
        let b = chunk.get(1).copied().unwrap_or(0) as usize;
        let c = chunk.get(2).copied().unwrap_or(0) as usize;
        out.push(DIGITS[a >> 2] as char);
        out.push(DIGITS[((a & 3) << 4) | (b >> 4)] as char);
        out.push(if chunk.len() > 1 {
            DIGITS[((b & 15) << 2) | (c >> 6)] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            DIGITS[c & 63] as char
        } else {
            '='
        });
    }
    out
}

/// The OSC 52 escape sequence that asks the terminal to copy `text`, wrapped
/// for tmux passthrough when `tmux` is true.
pub fn osc52(text: &str, tmux: bool) -> io::Result<String> {
    within_cap(text)?;
    let sequence = format!("\x1b]52;c;{}\x07", base64(text.as_bytes()));
    Ok(if tmux {
        format!("\x1bPtmux;{}\x1b\\", sequence.replace('\x1b', "\x1b\x1b"))
    } else {
        sequence
    })
}

/// Copy `text`: with pbcopy on a local Mac, otherwise through OSC 52 when the
/// terminal is interactive. Returns the method used.
pub fn copy(text: &str, interactive: bool) -> io::Result<&'static str> {
    within_cap(text)?;
    if std::env::var_os("SSH_TTY").is_none() && std::env::var_os("SSH_CONNECTION").is_none() {
        match Command::new("pbcopy")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(mut child) => {
                let written = child.stdin.take().unwrap().write_all(text.as_bytes());
                let status = child.wait()?;
                written?;
                return if status.success() {
                    Ok("pbcopy")
                } else {
                    Err(io::Error::other("Copy failed: pbcopy did not succeed."))
                };
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => (),
            Err(error) => return Err(error),
        }
    }
    if !interactive {
        return Err(io::Error::other(
            "Copy via OSC 52 requires an interactive terminal.",
        ));
    }
    let mut out = io::stdout().lock();
    out.write_all(osc52(text, std::env::var_os("TMUX").is_some())?.as_bytes())?;
    out.flush()?;
    Ok("OSC 52")
}
