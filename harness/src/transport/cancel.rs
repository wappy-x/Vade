//! Request-scoped engine cancellation; no control is sent to unknown servers.
use super::*;
pub(super) struct Control { endpoint: Endpoint, body: String }
pub(super) fn prepare(endpoint: &Endpoint, body: &[u8], enabled: bool) -> io::Result<(Vec<u8>, Option<Control>)> {
    if !enabled { return Ok((body.to_vec(), None)); }
    let endpoint = crate::contract::sibling(endpoint, "cancel").ok_or_else(|| invalid("cancel route unavailable"))?;
    let raw = std::str::from_utf8(body).map_err(|_| invalid("request encoding"))?;
    let existing = json::view(raw)?.get("cancel").and_then(|v| v.as_str()).map(|v| v.into_owned());
    let key = match &existing {
        Some(key) if key.len() == 64 && key.bytes().all(|b| b.is_ascii_hexdigit()) => key.clone(),
        Some(_) => return Err(invalid("invalid cancel capability")),
        None => {
            let mut bytes = [0u8; 32];
            std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
            bytes.iter().map(|byte| format!("{byte:02x}")).collect()
        }
    };
    let control = format!("{{\"cancel\":\"{key}\"}}");
    let body = if existing.is_some() { body.to_vec() } else {
        let prefix = raw.trim_end().strip_suffix('}').ok_or_else(|| invalid("request object"))?;
        format!("{prefix},{}", &control[1..]).into_bytes()
    };
    Ok((body, Some(Control { endpoint, body: control })))
}
/// Monitor independently of the generation socket, including before headers.
/// Join before returning so a subsequent steering request cannot overtake it.
pub(super) fn run<T>(control: Option<Control>, cancel: Option<&AtomicBool>, request: impl FnOnce() -> io::Result<T>) -> (io::Result<T>, Option<bool>) {
    std::thread::scope(|scope| {
        let (done, finished) = std::sync::mpsc::channel::<()>();
        let watcher = control.zip(cancel).map(|(control, flag)| scope.spawn(move || loop {
            if flag.load(Ordering::Relaxed) {
                return Some((|| -> io::Result<bool> {
                    let mut response = request_method_pooled_deadline("POST", &control.endpoint, control.body.as_bytes(),
                        None, None, &ConnectionPool::default(), Some(Instant::now() + Duration::from_secs(2)))?;
                    let status = response.status;
                    let mut bytes = Vec::new();
                    response.body.by_ref().take(1025).read_to_end(&mut bytes)?;
                    Ok(status == 200 && bytes.len() <= 1024 && json::parse_bytes(&bytes)?
                        .get("cancelled") == Some(&json::Value::Bool(true)))
                })().unwrap_or(false));
            }
            match finished.recv_timeout(Duration::from_millis(5)) {
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => (),
                _ if flag.load(Ordering::Relaxed) => continue,
                _ => return None,
            }
        }));
        let result = request();
        drop(done);
        (result, watcher.and_then(|worker| worker.join().ok().flatten()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::mpsc;

    #[test]
    fn stalled_control_response_cannot_hold_the_cancel_join() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = Endpoint::parse(&format!("http://{}/v1/cancel", listener.local_addr().unwrap())).unwrap();
        let (release, stall) = mpsc::channel();
        let (received, receipt) = mpsc::channel();
        let server = std::thread::spawn(move || {
            let until = Instant::now() + Duration::from_secs(5);
            let socket = loop {
                match listener.accept() {
                    Ok((socket, _)) => break socket,
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < until => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("control accept failed: {error}"),
                }
            };
            socket.set_nonblocking(false).unwrap();
            socket.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
            let mut reader = BufReader::new(socket);
            let head = read_head(&mut reader).unwrap();
            assert_eq!(head.first, "POST /v1/cancel HTTP/1.1");
            let mut bytes = Vec::new();
            BodyReader::new(&mut reader, &head, true, 1024).unwrap().read_to_end(&mut bytes).unwrap();
            received.send(bytes).unwrap();
            let _ = stall.recv_timeout(Duration::from_secs(5));
        });
        let (finished, result) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let flag = AtomicBool::new(true);
            let control = Control { endpoint, body: "{\"cancel\":\"test\"}".into() };
            finished.send(run(Some(control), Some(&flag), || Ok(()))).unwrap();
        });
        let observed = result.recv_timeout(Duration::from_secs(4));
        // Unblock the fixture before any assertion, and never wait without a
        // deadline for a cancellation worker that may contain a regression.
        let _ = release.send(());
        let request = receipt.recv_timeout(Duration::from_secs(1));
        server.join().unwrap();
        let (generation, confirmed) = observed.expect("cancel join exceeded its two-second control budget");
        worker.join().unwrap();
        assert_eq!(request.unwrap(), b"{\"cancel\":\"test\"}");
        assert!(generation.is_ok());
        assert_eq!(confirmed, Some(false));
    }
}
