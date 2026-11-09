//! Shared first-party bounded stdio transport for delegated agents. Protocol
//! methods, permission policy, and session ownership stay in each adapter.
use super::{Cleanup, MAX_LINE, POLL, check_cancel, invalid, timed_out};
use crate::json::{self, Value};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver, SyncSender},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
#[cfg(unix)]
use std::{os::fd::OwnedFd, os::unix::net::UnixStream};
const OUTPUT_QUEUE: usize = 8;
struct WriteJob {
    bytes: Vec<u8>,
    ack: SyncSender<io::Result<()>>,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
}

pub(super) struct Connection {
    child: Option<Child>,
    outgoing: Option<SyncSender<WriteJob>>,
    incoming: Receiver<io::Result<Vec<u8>>>,
    reader: Option<JoinHandle<()>>,
    writer: Option<JoinHandle<()>>,
    io_stop: Arc<AtomicBool>,
    cleanup: Option<Cleanup>,
}
/// Read one UTF-8 JSONL record without an unbounded read_line allocation.
fn json_line<R: BufRead>(reader: &mut R) -> io::Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let bytes = reader.fill_buf()?;
        if bytes.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Err(invalid(
                    "delegated child ended with an incomplete JSONL record",
                ))
            };
        }
        let take = bytes
            .iter()
            .position(|&b| b == b'\n')
            .map_or(bytes.len(), |i| i + 1);
        if line.len().saturating_add(take) > MAX_LINE {
            return Err(invalid("delegated child JSONL record exceeds limit"));
        }
        line.extend_from_slice(&bytes[..take]);
        reader.consume(take);
        if line.last() == Some(&b'\n') {
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if line.is_empty() {
                return Err(invalid("delegated child emitted an empty JSONL record"));
            }
            return Ok(Some(line));
        }
    }
}

fn read_output<R: Read>(output: R, queue: SyncSender<io::Result<Vec<u8>>>) {
    let mut reader = BufReader::new(output);
    loop {
        let record = match json_line(&mut reader) {
            Ok(Some(record)) => Ok(record),
            Ok(None) => Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "delegated child stdout closed",
            )),
            Err(error) => Err(error),
        };
        let terminal = record.is_err();
        // Bounded backpressure preserves bursts instead of dropping events.
        if queue.send(record).is_err() {
            break;
        }
        if terminal {
            break;
        }
    }
}

fn write_input<W: Write>(mut input: W, jobs: Receiver<WriteJob>) {
    while let Ok(job) = jobs.recv() {
        let result = write_record(&mut input, &job.bytes, job.deadline, &job.cancel);
        let failed = result.is_err();
        let _ = job.ack.try_send(result);
        if failed {
            break;
        }
    }
}

fn write_record<W: Write>(input: &mut W, mut bytes: &[u8], deadline: Instant, cancel: &AtomicBool) -> io::Result<()> {
    while !bytes.is_empty() {
        check_cancel(cancel)?;
        if Instant::now() >= deadline {
            return Err(timed_out("delegated child write deadline exceeded"));
        }
        match input.write(bytes) {
            Ok(0) => return Err(io::Error::new(io::ErrorKind::WriteZero, "delegated child input closed")),
            Ok(count) => bytes = &bytes[count..],
            Err(error) if matches!(error.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::Interrupted) => continue,
            Err(error) => return Err(error),
        }
    }
    // A complete record can already have started remote work. Its receipt must
    // win over a late cancel so the adapter can send the protocol interruption.
    input.flush()
}

/// std Unix socket pairs serve as child stdio, allowing bounded I/O without
/// unsafe code or a platform crate. A descendant retaining a descriptor cannot
/// prevent the parent's I/O workers from stopping.
#[cfg(unix)]
struct Pipe {
    socket: UnixStream,
    stop: Arc<AtomicBool>,
}

#[cfg(unix)]
impl Read for Pipe {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.stop.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    "delegated child I/O stopped",
                ));
            }
            match self.socket.read(buffer) {
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                            | io::ErrorKind::Interrupted
                    ) =>
                {
                    continue;
                }
                result => return result,
            }
        }
    }
}

#[cfg(unix)]
impl Write for Pipe {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.stop.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "delegated child I/O stopped",
            ));
        }
        // Return bounded socket timeouts to the record writer. It must check
        // this job's cancellation before it retries a partial record.
        self.socket.write(buffer)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.socket.flush()
    }
}

impl Connection {
    pub(super) fn spawn(command: Command) -> io::Result<Self> {
        #[cfg(unix)]
        {
            Self::spawn_unix(command)
        }
        #[cfg(not(unix))]
        {
            let _ = command;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "bounded delegated stdio requires Unix",
            ))
        }
    }
    #[cfg(unix)]
    fn spawn_unix(mut command: Command) -> io::Result<Self> {
        let (input, child_input) = UnixStream::pair()?;
        let (output, child_output) = UnixStream::pair()?;
        input.set_write_timeout(Some(POLL))?;
        output.set_read_timeout(Some(POLL))?;
        let child_input: OwnedFd = child_input.into();
        let child_output: OwnedFd = child_output.into();
        let io_stop = Arc::new(AtomicBool::new(false));
        let input = Pipe {
            socket: input,
            stop: Arc::clone(&io_stop),
        };
        let output = Pipe {
            socket: output,
            stop: Arc::clone(&io_stop),
        };
        command
            .stdin(Stdio::from(child_input))
            .stdout(Stdio::from(child_output))
            .stderr(Stdio::null());
        let mut child = command.spawn()?;
        drop(command);
        let (outgoing, writes) = mpsc::sync_channel(4);
        let (output_queue, incoming) = mpsc::sync_channel(OUTPUT_QUEUE);
        let reader = match thread::Builder::new()
            .name("delegated-stdout".into())
            .spawn(move || read_output(output, output_queue))
        {
            Ok(reader) => reader,
            Err(error) => {
                io_stop.store(true, Ordering::Release);
                drop(incoming);
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        };
        let writer = match thread::Builder::new()
            .name("delegated-stdin".into())
            .spawn(move || write_input(input, writes))
        {
            Ok(writer) => writer,
            Err(error) => {
                io_stop.store(true, Ordering::Release);
                drop(incoming);
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return Err(error);
            }
        };
        Ok(Self {
            child: Some(child),
            outgoing: Some(outgoing),
            incoming,
            reader: Some(reader),
            writer: Some(writer),
            io_stop,
            cleanup: None,
        })
    }
    pub(super) fn is_closed(&self) -> bool {
        self.child.is_none()
    }
    pub(super) fn cleanup(&self) -> Option<Cleanup> {
        self.cleanup
    }
    #[cfg(test)]
    pub(super) fn stopped(&self) -> bool {
        self.child.is_none() && self.reader.is_none() && self.writer.is_none()
    }
    #[cfg(test)]
    pub(super) fn child_id(&self) -> Option<u32> {
        self.child.as_ref().map(Child::id)
    }
    pub(super) fn send(
        &mut self,
        value: Value,
        deadline: Instant,
        cancel: Option<&AtomicBool>,
    ) -> io::Result<()> {
        if let Some(cancel) = cancel {
            check_cancel(cancel)?;
        }
        let mut bytes = json::stringify(&value)?.into_bytes();
        if bytes.len() >= MAX_LINE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "delegated child outgoing JSONL record exceeds limit",
            ));
        }
        bytes.push(b'\n');
        let (ack, result) = mpsc::sync_channel(1);
        let write_cancel = Arc::new(AtomicBool::new(false));
        self.outgoing
            .as_ref()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "delegated child client is closed",
                )
            })?
            .try_send(WriteJob { bytes, ack, deadline, cancel: Arc::clone(&write_cancel) })
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "delegated child write queue unavailable",
                )
            })?;
        loop {
            if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                // The writer owns the outcome after enqueue. A completed
                // record needs protocol cancellation; a partial one must stop.
                write_cancel.store(true, Ordering::Relaxed);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(timed_out("delegated child write deadline exceeded"));
            }
            match result.recv_timeout(POLL.min(remaining)) {
                Ok(result) => return result,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "delegated child writer stopped",
                    ));
                }
            }
        }
    }

    pub(super) fn receive(&mut self, wait: Duration) -> io::Result<Option<(Value, usize)>> {
        match self.incoming.recv_timeout(wait) {
            Ok(Ok(bytes)) => {
                let value = json::parse_bytes(&bytes)?;
                if !matches!(value, Value::Object(_)) {
                    return Err(invalid("delegated message must be an object"));
                }
                Ok(Some((value, bytes.len())))
            }
            Ok(Err(error)) => Err(error),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "delegated child output reader stopped",
            )),
        }
    }

    pub(super) fn shutdown(&mut self) {
        self.io_stop.store(true, Ordering::Release);
        self.outgoing.take();
        let (sender, closed) = mpsc::sync_channel(0);
        drop(sender);
        self.incoming = closed;
        let mut report = self.cleanup.unwrap_or(Cleanup {
            child_reaped: false,
            io_workers_joined: true,
        });
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            report.child_reaped = child.wait().is_ok();
            if !report.child_reaped {
                self.child = Some(child);
            }
        }
        // Closing the queue releases producer backpressure; socket timeouts
        // release workers even if an external descendant retains child stdio.
        for worker in [self.reader.take(), self.writer.take()]
            .into_iter()
            .flatten()
        {
            report.io_workers_joined &= worker.join().is_ok();
        }
        self.cleanup = Some(report);
    }
}
impl Drop for Connection {
    fn drop(&mut self) {
        self.shutdown();
    }
}
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    #[test]
    fn queued_write_resolves_before_cancellation() {
        let (outgoing, writes) = mpsc::sync_channel(1);
        let (_, incoming) = mpsc::sync_channel(1);
        let mut connection = Connection {
            child: None, outgoing: Some(outgoing), incoming, reader: None,
            writer: None, io_stop: Arc::new(AtomicBool::new(false)), cleanup: None,
        };
        let cancel = AtomicBool::new(true);
        let value = json::parse(r#"{"method":"session/prompt"}"#).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        assert_eq!(connection.send(value.clone(), deadline, Some(&cancel)).unwrap_err().kind(), io::ErrorKind::ConnectionAborted);
        assert!(matches!(writes.try_recv(), Err(mpsc::TryRecvError::Empty)));
        cancel.store(false, Ordering::Relaxed);
        thread::scope(|scope| {
            let (finished, result) = mpsc::sync_channel(1);
            let cancel = &cancel;
            scope.spawn(move || { let _ = finished.send(connection.send(value, deadline, Some(cancel))); });
            let job = writes.recv_timeout(Duration::from_secs(2)).unwrap();
            // The peer can read this prompt before the writer sends its receipt.
            // Cancellation must not abandon that accepted write as if unsent.
            cancel.store(true, Ordering::Relaxed);
            let before_ack = result.recv_timeout(POLL * 2);
            let _ = job.ack.send(Ok(()));
            let waiting = matches!(before_ack, Err(mpsc::RecvTimeoutError::Timeout));
            let result = if waiting { result.recv_timeout(Duration::from_secs(2)).unwrap() } else { before_ack.unwrap() };
            assert!(waiting, "accepted write returned before its receipt: {result:?}");
            result.unwrap();
        });
    }

    #[test]
    fn cancellation_distinguishes_complete_and_partial_records() {
        struct Writer<'a> { cancel: &'a AtomicBool, limit: usize, bytes: Vec<u8> }
        impl Write for Writer<'_> {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                let count = bytes.len().min(self.limit);
                self.bytes.extend_from_slice(&bytes[..count]);
                self.cancel.store(true, Ordering::Relaxed);
                Ok(count)
            }
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }
        let record = b"{\"method\":\"session/prompt\"}\n";
        for limit in [record.len(), record.len() - 1] {
            let cancel = AtomicBool::new(false);
            let mut writer = Writer { cancel: &cancel, limit, bytes: Vec::new() };
            let result = write_record(&mut writer, record, Instant::now() + Duration::from_secs(2), &cancel);
            assert_eq!(writer.bytes, record[..limit]);
            if limit == record.len() { result.unwrap(); }
            else { assert_eq!(result.unwrap_err().kind(), io::ErrorKind::ConnectionAborted); }
        }
    }

    #[test]
    fn jsonl_fragmentation_and_size_limit() {
        let data = "{\"text\":\"é🦀\"}\r\n";
        let mut reader = BufReader::with_capacity(1, data.as_bytes());
        assert_eq!(
            json::parse_bytes(&json_line(&mut reader).unwrap().unwrap())
                .unwrap()
                .get("text")
                .and_then(Value::as_str),
            Some("é🦀")
        );
        assert!(json_line(&mut BufReader::new(&b"{\"partial\":"[..])).is_err());
        assert!(
            json_line(&mut BufReader::new(io::Cursor::new(vec![
                b'x';
                MAX_LINE + 1
            ])))
            .is_err()
        );
    }

    #[test]
    fn stopped_stdio_workers_exit_even_when_peer_descriptors_remain_open() {
        let (read_socket, _read_peer) = UnixStream::pair().unwrap();
        let (write_socket, _write_peer) = UnixStream::pair().unwrap();
        read_socket.set_read_timeout(Some(POLL)).unwrap();
        write_socket.set_write_timeout(Some(POLL)).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let read_stop = Arc::clone(&stop);
        let write_stop = Arc::clone(&stop);
        let reader = thread::spawn(move || {
            Pipe {
                socket: read_socket,
                stop: read_stop,
            }
            .read(&mut [0; 1])
        });
        let writer = thread::spawn(move || {
            let mut input = Pipe {
                socket: write_socket,
                stop: write_stop,
            };
            write_record(&mut input, &vec![0; MAX_LINE * 2], Instant::now() + Duration::from_secs(5), &AtomicBool::new(false))
        });
        thread::sleep(Duration::from_millis(75));
        let start = Instant::now();
        stop.store(true, Ordering::Release);
        assert_eq!(
            reader.join().unwrap().unwrap_err().kind(),
            io::ErrorKind::ConnectionAborted
        );
        assert_eq!(
            writer.join().unwrap().unwrap_err().kind(),
            io::ErrorKind::ConnectionAborted
        );
        assert!(start.elapsed() < Duration::from_secs(2));
    }
}
