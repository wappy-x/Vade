//! Astra review of 7d418cd, BLOCK 1: `vade read` keeps parser storage within
//! the conversion budget. Each input is one that kept tens of MB of parser
//! storage before the fix, at a char cap of 1. The test measures the largest
//! resident memory and compares it with the same read of a small file. On
//! macOS it runs `vade read` under `/usr/bin/time -l`; on Linux it reads the
//! file in this process and takes the peak from `/proc/self/status` after it
//! resets the peak. When the measurement is not possible the test fails: it
//! never passes without measuring (Astra review of 8f038e3, finding 5).
#![cfg(any(target_os = "macos", target_os = "linux"))]

/// More growth than this over the small file means storage is not bounded.
/// Each input is at most 2.25 MB; reading it holds the bytes and one decoded copy.
const MAX_GROWTH: u64 = 16 * 1024 * 1024;

/// The largest resident memory, in bytes, of `vade read --max-chars 1 <file>`.
#[cfg(target_os = "macos")]
fn peak(file: &std::path::Path) -> u64 {
    let output = std::process::Command::new("/usr/bin/time")
        .arg("-l")
        .arg(env!("CARGO_BIN_EXE_vade"))
        .args(["read", "--max-chars", "1"])
        .arg(file)
        .env("LC_ALL", "C")
        .env("VADE_HOME", "relative-invalid")
        .env("VADE_CONFIG", "")
        .output()
        .expect("/usr/bin/time is part of macOS; the test needs it");
    let report = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{report}");
    let line = report.lines().find(|l| l.contains("maximum resident set size"));
    let number = line.and_then(|l| l.split_whitespace().find_map(|w| w.parse::<u64>().ok()));
    number.unwrap_or_else(|| panic!("no maximum resident set size in the /usr/bin/time report:\n{report}"))
}

/// The largest resident memory, in bytes, while this process reads the file
/// as `vade read --max-chars 1` does.
#[cfg(target_os = "linux")]
fn peak(file: &std::path::Path) -> u64 {
    use vade_harness::readers::{Caps, read};
    let bytes = std::fs::read(file).unwrap();
    // Writing 5 resets the peak to the current size (Linux 4.0 and later).
    std::fs::write("/proc/self/clear_refs", "5").expect("the test resets the peak in /proc/self/clear_refs");
    let result = read(&bytes, &file.to_string_lossy(), &Caps { max_chars: 1, ..Caps::default() });
    let status = std::fs::read_to_string("/proc/self/status").expect("the test reads the peak from /proc/self/status");
    assert!(result.is_ok(), "{result:?}");
    let line = status.lines().find(|l| l.starts_with("VmHWM:"));
    let kib = line.and_then(|l| l.split_whitespace().find_map(|w| w.parse::<u64>().ok()));
    kib.unwrap_or_else(|| panic!("no VmHWM line in /proc/self/status:\n{status}")) * 1024
}

#[test]
fn read_keeps_parser_storage_within_the_budget() {
    let dir = std::env::temp_dir().join(format!("vade-read-memory-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let small = dir.join("small.txt");
    std::fs::write(&small, "hello\n").unwrap();
    let base = peak(&small);
    let inputs: [(&str, Vec<u8>); 7] = [
        ("fields.csv", vec![b','; 1_000_000]),
        ("attributes.html", format!("<p {}>x</p>", "a ".repeat(500_000)).into_bytes()),
        ("rows.html", format!("<table>{}</table>", "<tr></tr>".repeat(125_000)).into_bytes()),
        ("lists.html", format!("{}{}", "<ul>".repeat(250_000), "</ul>".repeat(250_000)).into_bytes()),
        ("groups.rtf", format!("{{\\rtf1 {}{}}}", "{".repeat(1_000_000), "}".repeat(1_000_000)).into_bytes()),
        ("text.html", format!("<p>{}</p>", "y &amp; ".repeat(250_000)).into_bytes()),
        ("values.json", format!("[{}0]", "0,".repeat(130_000)).into_bytes()),
    ];
    for (name, bytes) in inputs {
        let file = dir.join(name);
        std::fs::write(&file, bytes).unwrap();
        let used = peak(&file);
        eprintln!("{name}: {used} bytes, small file {base} bytes");
        assert!(used.saturating_sub(base) < MAX_GROWTH, "{name}: {used} bytes, small file {base} bytes");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
