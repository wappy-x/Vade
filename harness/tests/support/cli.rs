//! Bounded first-party CLI driver. Commands that depend on completion wait for
//! an observed receipt, never a sleep or serial stdin consumption assumption.
use std::io::{Read, Write};
use std::process::{Child, Command, Output, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub struct Cli {
    child: Child,
    incoming: Receiver<(bool, Vec<u8>)>,
    readers: Vec<JoinHandle<()>>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}
impl Cli {
    pub fn start(command: &mut Command) -> Self {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let (tx, incoming) = mpsc::channel();
        let readers = [
            (false, Box::new(stdout) as Box<dyn Read + Send>),
            (true, Box::new(stderr)),
        ]
        .into_iter()
        .map(|(err, mut stream)| {
            let tx = tx.clone();
            thread::spawn(move || {
                let mut buffer = [0; 4096];
                let mut total = 0;
                loop {
                    let n = stream.read(&mut buffer).unwrap();
                    if n == 0 {
                        break;
                    }
                    total += n;
                    assert!(total <= 2 * 1024 * 1024, "fixture output exceeded limit");
                    if tx.send((err, buffer[..n].to_vec())).is_err() {
                        break;
                    }
                }
            })
        })
        .collect();
        Self {
            child,
            incoming,
            readers,
            stdout: Vec::new(),
            stderr: Vec::new(),
        }
    }
    pub fn send(&mut self, text: &str) {
        self.child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(text.as_bytes())
            .unwrap();
    }
    pub fn until(&mut self, stderr: bool, needle: &str) {
        let start = if stderr {
            self.stderr.len()
        } else {
            self.stdout.len()
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            self.receive(deadline);
            let output = if stderr { &self.stderr } else { &self.stdout };
            if output[start..]
                .windows(needle.len())
                .any(|bytes| bytes == needle.as_bytes())
            {
                return;
            }
        }
    }
    fn receive(&mut self, deadline: Instant) {
        let (err, bytes) = self
            .incoming
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .expect("CLI did not emit the expected receipt before the deadline");
        if err {
            self.stderr.extend(bytes);
        } else {
            self.stdout.extend(bytes);
        }
    }
    pub fn finish(mut self) -> Output {
        drop(self.child.stdin.take());
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match self.incoming.recv_timeout(Duration::from_millis(10)) {
                Ok((true, bytes)) => self.stderr.extend(bytes),
                Ok((false, bytes)) => self.stdout.extend(bytes),
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => (),
            }
            assert!(Instant::now() < deadline, "CLI cleanup exceeded deadline");
        }
        while self.child.try_wait().unwrap().is_none() {
            assert!(Instant::now() < deadline, "CLI exit exceeded deadline");
            thread::sleep(Duration::from_millis(1));
        }
        for reader in self.readers.drain(..) {
            reader.join().unwrap();
        }
        Output {
            status: self.child.wait().unwrap(),
            stdout: std::mem::take(&mut self.stdout),
            stderr: std::mem::take(&mut self.stderr),
        }
    }
}
impl Drop for Cli {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
    }
}
