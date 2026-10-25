//! Fallible public output. Callers unwind owned work before main selects an exit code.
use std::{cell::Cell, fmt, io::{self, Write}};

thread_local! {
    static FAILURE: Cell<Option<io::ErrorKind>> = const { Cell::new(None) };
    static OPERATION_FAILED: Cell<bool> = const { Cell::new(false) };
}

/// The error kind of the first failed write to standard output, if any.
pub fn failure() -> Option<io::ErrorKind> { FAILURE.get() }
/// Mark the command as failed, so a closed pipe cannot turn it into success.
pub fn operation_failed() { OPERATION_FAILED.set(true); }
/// Whether the only problem was a reader closing the pipe (as `| head` does),
/// which exits 0.
pub fn broken_pipe_success() -> bool {
    failure() == Some(io::ErrorKind::BrokenPipe) && !OPERATION_FAILED.get()
}
/// A cleanup failure must remain a failing command even if stdout closed first.
pub fn cleanup_failed() { operation_failed(); FAILURE.set(None); }

fn write_to(output: &mut dyn Write, args: fmt::Arguments<'_>, newline: bool) -> io::Result<()> {
    output.write_fmt(args)?;
    if newline { output.write_all(b"\n")?; }
    output.flush()
}

thread_local! { static BUFFER: std::cell::RefCell<Option<Vec<u8>>> = const { std::cell::RefCell::new(None) }; }

/// Collect this thread's standard output instead of printing it, so concurrent
/// workers can be printed afterwards in a fixed order.
pub fn buffered<T>(run: impl FnOnce() -> T) -> (T, String) {
    BUFFER.set(Some(Vec::new()));
    let value = run();
    let bytes = BUFFER.take().unwrap_or_default();
    (value, String::from_utf8_lossy(&bytes).into_owned())
}

fn write(args: fmt::Arguments<'_>, newline: bool) -> Result<(), String> {
    if let Some(result) = BUFFER.with_borrow_mut(|buffer| buffer.as_mut().map(|bytes| write_to(bytes, args, newline))) {
        return result.map_err(|_| "Cannot buffer standard output.".into());
    }
    #[cfg(test)]
    let result = TEST_WRITER.with_borrow_mut(|writer| match writer {
        Some(writer) => write_to(writer.as_mut(), args, newline),
        None => write_to(&mut io::stdout().lock(), args, newline),
    });
    #[cfg(not(test))]
    let result = write_to(&mut io::stdout().lock(), args, newline);
    result.map_err(|error| {
        FAILURE.set(Some(error.kind()));
        "Cannot write standard output. Check the output destination, permissions and available space, then retry.".into()
    })
}

/// Write to standard output and flush; a failed write is an error to unwind.
pub fn text(args: fmt::Arguments<'_>) -> Result<(), String> { write(args, false) }
/// [`text`] followed by a newline.
pub fn line(args: fmt::Arguments<'_>) -> Result<(), String> { write(args, true) }

/// Legacy test/report commands own their exit codes and no launcher resources.
/// Preserve command failures; output errors can only turn success into failure.
pub fn report(args: fmt::Arguments<'_>, newline: bool) { let _ = write(args, newline); }
/// The process exit code for a command that returned `code`, after any output
/// failure.
pub fn exit_code(code: i32) -> i32 {
    if code != 0 { return code; }
    match failure() {
        Some(io::ErrorKind::BrokenPipe) => 0,
        Some(_) => { stderr_line(format_args!("Cannot write standard output. Check the output destination and retry.")); 1 }
        None => code,
    }
}

/// Best-effort write to standard error.
pub fn stderr(args: fmt::Arguments<'_>) { let _ = io::stderr().lock().write_fmt(args); }
/// Best-effort line on standard error.
pub fn stderr_line(args: fmt::Arguments<'_>) {
    let _ = writeln!(io::stderr().lock(), "{args}");
}

#[cfg(test)]
thread_local! { static TEST_WRITER: std::cell::RefCell<Option<Box<dyn Write>>> = const { std::cell::RefCell::new(None) }; }

/// Send this thread's standard output to `writer` while `run` runs (tests only).
#[cfg(test)]
pub fn with_writer<T>(writer: impl Write + 'static, run: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) { TEST_WRITER.with_borrow_mut(|writer| *writer = None); FAILURE.set(None); OPERATION_FAILED.set(false); }
    }
    let _reset = Reset;
    TEST_WRITER.with_borrow_mut(|slot| { assert!(slot.is_none()); *slot = Some(Box::new(writer)); });
    run()
}

/// Test writers for [`with_writer`].
#[cfg(test)]
pub mod testing {
    use super::*;
    use std::{cell::RefCell, rc::Rc};
    /// Collects output and can inject a write or flush failure.
    #[derive(Clone, Default)]
    pub struct Capture {
        pub bytes: Rc<RefCell<Vec<u8>>>,
        pub fail_on: Option<(&'static str, io::ErrorKind)>,
        pub flush_error: Option<io::ErrorKind>,
    }
    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let mut output = self.bytes.borrow_mut();
            output.extend_from_slice(bytes);
            if let Some((needle, kind)) = self.fail_on {
                if String::from_utf8_lossy(&output).contains(needle) {
                    return Err(io::Error::new(kind, "injected private output failure"));
                }
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.flush_error.map_or(Ok(()), |kind| Err(io::Error::new(kind, "injected flush failure")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{testing::Capture, *};
    #[test]
    fn write_and_flush_failures_preserve_exit_classification() {
        for kind in [io::ErrorKind::BrokenPipe, io::ErrorKind::PermissionDenied, io::ErrorKind::StorageFull, io::ErrorKind::Other] {
            for flush in [false, true] {
                let writer = Capture { fail_on: (!flush).then_some(("output", kind)), flush_error: flush.then_some(kind), ..Capture::default() };
                with_writer(writer, || {
                    let error = text(format_args!("output")).unwrap_err();
                    assert!(error.contains("standard output") && error.contains("retry"));
                    assert!(!error.contains("private"));
                    assert_eq!(failure(), Some(kind));
                    assert_eq!(exit_code(7), 7);
                    assert_eq!(exit_code(0), if kind == io::ErrorKind::BrokenPipe { 0 } else { 1 });
                });
            }
        }
    }
}
