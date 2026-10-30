//! `vade read <path>`: print a local file as Markdown through the harness
//! document readers. No engine, network or config is needed.
use vade_harness::readers::{self, Caps, ReadError};

pub fn help() -> String {
    crate::style::Page {
        summary: "Print a local file as Markdown, with a header line.",
        usage: &["vade read <PATH> [--max-chars N]"],
        options: &[("--max-chars N", "Characters of Markdown to keep, 1..2000000 (default 2000000)")],
        examples: &["vade read notes.txt", "vade read report.html --max-chars 20000", "vade read table.csv"],
        notes: &[
            "Reads text (UTF-8, UTF-16, Latin-1), Markdown, HTML, CSV, TSV, JSON, YAML, TOML, XML, RTF, PDF, DOCX, XLSX, PPTX, EPUB, ODT, ODS, ODP and source files, and gives the size of PNG, JPEG, GIF and WebP images. PDF, office and EPUB output has a marker before each page, sheet, slide or chapter. Other known formats print one line that names the format; other binary files print one line that says so.",
            "The first line is <!-- vade read: NAME | FORMAT | pages N | chars C | skipped: ... -->. Files over 50 MiB are refused.",
            "Exit 0 when text was printed, 1 when the file cannot be read, 2 for a usage error. See docs/reading-files.md.",
        ],
    }
    .render(crate::style::color(crate::style::Stream::Out))
}

/// The exit code: 0 printed, 1 unreadable, 2 usage.
pub fn run(args: &[String]) -> i32 {
    let (path, caps) = match parse(args) {
        Ok(parsed) => parsed,
        Err(message) => {
            crate::output::stderr_line(format_args!("error: {message}"));
            return 2;
        }
    };
    match markdown(&path, &caps) {
        // A reader that closes the pipe early (`| head`) is success, as for
        // every other command; any other output error fails.
        Ok(text) => match crate::output::text(format_args!("{text}")) {
            Ok(()) => 0,
            Err(_) if crate::output::broken_pipe_success() => 0,
            Err(error) => {
                crate::output::stderr_line(format_args!("error: {error}"));
                1
            }
        },
        Err(message) => {
            crate::output::stderr_line(format_args!("error: {}: {message}", crate::text::argument(&path)));
            1
        }
    }
}

fn parse(args: &[String]) -> Result<(String, Caps), String> {
    let mut caps = Caps::default();
    let mut path = None;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if arg == "--max-chars" {
            let value = args.get(index + 1).ok_or("Give --max-chars a value. Run vade read --help.")?;
            caps.max_chars = value
                .parse()
                .ok()
                .filter(|n| (1..=readers::MAX_CHARS).contains(n))
                .ok_or_else(|| format!("--max-chars must be 1..{}. Run vade read --help.", readers::MAX_CHARS))?;
            index += 2;
            continue;
        }
        if arg.starts_with('-') && arg != "-" || path.is_some() {
            return Err(format!("Unexpected argument {} for vade read. Run vade read --help.", crate::text::argument(arg)));
        }
        path = Some(arg.to_owned());
        index += 1;
    }
    Ok((path.ok_or("Give vade read a file path. Run vade read --help.")?, caps))
}

/// Why `metadata` is not a regular file, or `None` for a regular file.
fn not_a_file(metadata: &std::fs::Metadata) -> Option<String> {
    use std::os::unix::fs::FileTypeExt;
    let kind = metadata.file_type();
    let what = if kind.is_file() {
        return None;
    } else if kind.is_dir() {
        return Some("is a directory; give a file".into());
    } else if kind.is_fifo() {
        "named pipe"
    } else if kind.is_socket() {
        "socket"
    } else if kind.is_block_device() || kind.is_char_device() {
        "device"
    } else {
        "special file"
    };
    Some(format!("is not a regular file ({what}); give a file"))
}

fn markdown(path: &str, caps: &Caps) -> Result<String, String> {
    use std::io::Read;
    use std::os::unix::fs::OpenOptionsExt;
    // The type check comes before the open: opening a named pipe waits for a
    // writer, and opening a device can act on it. A symlink is followed to
    // its target. The open is nonblocking, so a pipe swapped in after the
    // check cannot hang either; the opened handle is checked again.
    let target = std::fs::symlink_metadata(path)
        .and_then(|m| if m.file_type().is_symlink() { std::fs::metadata(path) } else { Ok(m) })
        .map_err(|e| format!("cannot open the file ({})", e.kind()))?;
    if let Some(why) = not_a_file(&target) {
        return Err(why);
    }
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(crate::unix_flags::O_NONBLOCK | crate::unix_flags::O_CLOEXEC)
        .open(path)
        .map_err(|e| format!("cannot open the file ({})", e.kind()))?;
    let metadata = file.metadata().map_err(|e| format!("cannot read the file ({})", e.kind()))?;
    if let Some(why) = not_a_file(&metadata) {
        return Err(why);
    }
    if metadata.len() > caps.max_bytes as u64 {
        return Err(ReadError::TooLarge { bytes: metadata.len(), cap: caps.max_bytes }.to_string());
    }
    let mut bytes = Vec::new();
    file.take(caps.max_bytes as u64 + 1).read_to_end(&mut bytes).map_err(|e| format!("cannot read the file ({})", e.kind()))?;
    readers::read_markdown(&bytes, path, caps).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn parses_path_and_char_cap() {
        assert_eq!(parse(&args(&["a.txt"])).unwrap(), ("a.txt".into(), Caps::default()));
        let (_, caps) = parse(&args(&["--max-chars", "10", "a.txt"])).unwrap();
        assert_eq!(caps.max_chars, 10);
        for bad in [&[][..], &["a", "b"], &["--max-chars", "0", "a"], &["--max-chars"], &["--pages", "a"]] {
            assert!(parse(&args(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn reads_a_file_as_markdown() {
        let dir = std::env::temp_dir().join(format!("vade-read-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("t.csv");
        std::fs::write(&csv, "a,b\n1,2\n").unwrap();
        let text = markdown(csv.to_str().unwrap(), &Caps::default()).unwrap();
        assert_eq!(text, "<!-- vade read: t.csv | CSV | pages 1 | chars 34 | skipped: none -->\n| a | b |\n| --- | --- |\n| 1 | 2 |\n");
        let png = dir.join("i.png");
        std::fs::write(&png, b"\x89PNG\r\n\x1a\n").unwrap();
        assert_eq!(markdown(png.to_str().unwrap(), &Caps::default()).unwrap_err(), "damaged PNG image file: the PNG header is cut short; no text was read");
        let tiff = dir.join("i.tif");
        std::fs::write(&tiff, b"II*\0").unwrap();
        assert_eq!(markdown(tiff.to_str().unwrap(), &Caps::default()).unwrap_err(), "Vade does not read TIFF image files; no text was read");
        assert!(markdown(dir.to_str().unwrap(), &Caps::default()).unwrap_err().contains("directory"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Astra review of 075165a, BLOCK 2: a named pipe with no writer must not
    /// hang `vade read`. The read runs on a thread with a deadline; on a hang
    /// the test opens the pipe for writing to release it, then fails. That
    /// open does not block (Astra SHOULD-FIX 4 of 7d418cd): with no reader
    /// waiting it fails at once, and the test still fails on its assertion.
    #[test]
    fn a_named_pipe_is_refused_without_blocking() {
        use std::os::unix::fs::OpenOptionsExt;
        let dir = std::env::temp_dir().join(format!("vade-read-fifo-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fifo = dir.join("pipe.txt");
        let made = std::process::Command::new("mkfifo").arg(&fifo).status().is_ok_and(|s| s.success());
        assert!(made, "mkfifo is needed for this test");
        let path = fifo.to_str().unwrap().to_owned();
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::spawn(move || send.send(markdown(&path, &Caps::default())));
        let result = receive.recv_timeout(std::time::Duration::from_secs(5));
        if result.is_err() {
            let _ = std::fs::OpenOptions::new().write(true).custom_flags(crate::unix_flags::O_NONBLOCK).open(&fifo);
        }
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(result.expect("vade read blocked on a named pipe, or its thread panicked").unwrap_err(), "is not a regular file (named pipe); give a file");
        assert_eq!(markdown("/dev/null", &Caps::default()).unwrap_err(), "is not a regular file (device); give a file");
    }

    /// Astra SHOULD-FIX 11: a reader that closes the pipe (`| head`) is
    /// success; other output errors still fail.
    #[test]
    fn a_closed_output_pipe_exits_zero() {
        use crate::output::{testing::Capture, with_writer};
        let dir = std::env::temp_dir().join(format!("vade-read-pipe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("a.txt");
        std::fs::write(&file, "hello\n").unwrap();
        let path = file.to_str().unwrap();
        for (kind, code) in [(std::io::ErrorKind::BrokenPipe, 0), (std::io::ErrorKind::StorageFull, 1)] {
            let writer = Capture { fail_on: Some(("vade read", kind)), ..Capture::default() };
            assert_eq!(with_writer(writer, || run(&args(&[path]))), code, "{kind:?}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
