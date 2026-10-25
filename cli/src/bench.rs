//! Benchmark receipts belong in the working directory, not the installation.

/// `vade bench`: run vade-bench, writing to `./vade-bench.json` unless
/// `--output` (or `--out`) is given.
pub fn run(arguments: &[String], interrupted: fn() -> bool) -> Result<i32, String> {
    // Discovery and the development suite have their own argument handling.
    if arguments.first().is_some_and(|arg| arg == "--suite")
        || arguments.iter().any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        return vade_bench::run(arguments, interrupted);
    }
    let mut arguments: Vec<String> = arguments.iter()
        .map(|arg| if arg == "--out" { "--output".into() } else { arg.clone() }).collect();
    if arguments.windows(2).any(|pair| pair[0] == "--output" && pair[1].is_empty()) {
        return Err("missing value for --output".into());
    }
    if !arguments.iter().any(|arg| arg == "--output") {
        arguments.extend(["--output".into(), "./vade-bench.json".into()]);
    }
    let status = vade_bench::run(&arguments, interrupted)?;
    // With the server's own drafting policy, say where it chose plain decoding:
    // those rows measure plain speed, which is why a level can trail the one before.
    if !arguments.iter().any(|arg| arg == "--draft") {
        let output = arguments.windows(2).find(|pair| pair[0] == "--output").map(|pair| pair[1].clone());
        if let Some(note) = output.and_then(|path| std::fs::read_to_string(path).ok()).and_then(|receipt| plain_note(&receipt)) {
            write_note(&mut std::io::stdout().lock(), &note);
        }
    }
    Ok(status)
}

/// The receipt is saved and the status decided; a closed or failing stdout
/// changes neither, like the benchmark's own output (a broken pipe is fine).
fn write_note(out: &mut impl std::io::Write, note: &str) {
    let _ = writeln!(out, "{note}").and_then(|()| out.flush());
}

/// One line naming each concurrency where some replies reported plain
/// decoding (`sampling_echo.vade.draft`), or None when every reply drafted.
fn plain_note(receipt: &str) -> Option<String> {
    let mut levels = Vec::new();
    for arm in (0..).map(|i: usize| i.to_string()).take_while(|arm| crate::json::path_exists(receipt, &["arms", arm])) {
        for cell in (0..).map(|i: usize| i.to_string()).take_while(|cell| crate::json::path_exists(receipt, &["arms", &arm, "cells", cell])) {
            let Some(level) = crate::json::path_number(receipt, &at(&arm, &cell, &["concurrency"])) else { continue };
            let modes = (0..).map(|i: usize| i.to_string())
                .map_while(|request| crate::json::path_exists(receipt, &at(&arm, &cell, &["requests", &request]))
                    .then(|| crate::json::path_string(receipt, &at(&arm, &cell, &["requests", &request, "sampling_echo", "vade", "draft"]))))
                .flatten().collect::<Vec<_>>();
            let plain = modes.iter().filter(|mode| *mode == "plain").count();
            if plain > 0 { levels.push(format!("{plain} of {} replies at {level} chats", modes.len())); }
        }
    }
    (!levels.is_empty()).then(|| format!(
        "Drafting: the server answered without drafting (plain decoding) for {}; those rows measure plain speed. Each reply's mode is under sampling_echo in the receipt.",
        levels.join(", ")))
}

fn at<'a>(arm: &'a str, cell: &'a str, path: &[&'a str]) -> Vec<&'a str> {
    [&["arms", arm, "cells", cell][..], path].concat()
}

#[cfg(test)]
mod tests {
    #[test]
    fn plain_replies_are_named_per_concurrency() {
        let request = |mode: &str| format!(r#"{{"sampling_echo":{{"vade":{{"draft":"{mode}"}}}}}}"#);
        let cell = |level: u32, modes: &[&str]| format!(r#"{{"concurrency":{level},"requests":[{}]}}"#,
            modes.iter().map(|mode| request(mode)).collect::<Vec<_>>().join(","));
        let receipt = |cells: &[String]| format!(r#"{{"arms":[{{"cells":[{}]}}]}}"#, cells.join(","));
        assert_eq!(super::plain_note(&receipt(&[cell(1, &["strict"]), cell(4, &["strict"; 4])])), None);
        assert_eq!(super::plain_note(&receipt(&[cell(4, &["strict"; 4]), cell(8, &["plain", "plain", "strict", "plain"])])).unwrap(),
            "Drafting: the server answered without drafting (plain decoding) for 3 of 4 replies at 8 chats; those rows measure plain speed. Each reply's mode is under sampling_echo in the receipt.");
        assert_eq!(super::plain_note("not json"), None);
    }

    #[test]
    fn a_closed_or_failing_stdout_leaves_the_note_silent() {
        struct Closed(std::io::ErrorKind);
        impl std::io::Write for Closed {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> { Err(self.0.into()) }
            fn flush(&mut self) -> std::io::Result<()> { Err(self.0.into()) }
        }
        for kind in [std::io::ErrorKind::BrokenPipe, std::io::ErrorKind::Other] {
            super::write_note(&mut Closed(kind), "Drafting: note");
        }
        let mut written = Vec::new();
        super::write_note(&mut written, "Drafting: note");
        assert_eq!(written, b"Drafting: note\n");
    }
}
