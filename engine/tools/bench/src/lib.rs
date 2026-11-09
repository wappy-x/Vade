//! The one benchmark implementation used by `vade bench`.
pub mod fp8_reference;
pub mod norm_reference;
pub mod attention_reference;
pub mod gdn_reference;
pub mod first_tokens;
#[path = "main.rs"]
mod suite;
pub(crate) use suite::{Config, Result, http};
mod timed;

pub fn run(arguments: &[String], interrupted: fn() -> bool) -> Result<i32> {
    if arguments.first().is_some_and(|arg| arg == "--first-tokens") {
        first_tokens::run(&arguments[1..], interrupted)
    } else if arguments.first().is_some_and(|arg| arg == "--suite") {
        if arguments[1..]
            .iter()
            .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
        {
            write_help(&format!(
                "{}\nDevelopment workloads; see engine/tools/bench/README.md. Use the default\nvade bench protocol for current scoreboard receipts.\n",
                suite::HELP
            ))?;
            return Ok(0);
        }
        suite::run(&arguments[1..])?;
        Ok(0)
    } else {
        timed::run(arguments, interrupted)
    }
}

// Help is offline and may be piped into a consumer that exits early. Keep
// benchmark execution/output failures separate from this discovery path.
fn write_help(help: &str) -> Result<()> {
    use std::io::{self, Write};
    let mut output = io::stdout().lock();
    match output.write_all(help.as_bytes()).and_then(|()| output.flush()) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(_) => Err("Cannot write benchmark help. Check the output destination and retry.".into()),
    }
}

// The benchmark has already joined its workers before emitting its summary.
pub(crate) fn write_output(args: std::fmt::Arguments<'_>, newline: bool) -> Result<()> {
    use std::io::{self, Write};
    let mut output = io::stdout().lock();
    let result = output.write_fmt(args)
        .and_then(|()| if newline { output.write_all(b"\n") } else { Ok(()) })
        .and_then(|()| output.flush());
    match result {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(_) => Err("Cannot write benchmark output. Check the output destination and retry.".into()),
    }
}
pub(crate) fn stderr_line(args: std::fmt::Arguments<'_>) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr().lock(), "{args}");
}
