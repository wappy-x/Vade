//! A request deadline also bounds the wait for the operating system resolver.
use super::Endpoint;
use std::io;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::sync::mpsc;
use std::time::Instant;

pub(crate) fn addresses(
    endpoint: &Endpoint,
    deadline: Option<Instant>,
) -> io::Result<Vec<SocketAddr>> {
    if let Ok(ip) = endpoint.host.parse::<IpAddr>() {
        return Ok(vec![SocketAddr::new(ip, endpoint.port)]);
    }
    let host = endpoint.host.clone();
    let port = endpoint.port;
    let lookup = move || {
        (host.as_str(), port)
            .to_socket_addrs()
            .map(|addresses| addresses.take(16).collect())
    };
    match deadline {
        Some(deadline) => before(deadline, lookup),
        None => lookup(),
    }
}

fn timeout() -> io::Error {
    io::Error::new(
        io::ErrorKind::TimedOut,
        "HTTP request deadline exceeded during hostname resolution",
    )
}

fn before(
    deadline: Instant,
    lookup: impl FnOnce() -> io::Result<Vec<SocketAddr>> + Send + 'static,
) -> io::Result<Vec<SocketAddr>> {
    if Instant::now() >= deadline {
        return Err(timeout());
    }
    let (send, receive) = mpsc::sync_channel(1);
    // The OS lookup itself cannot be cancelled. Only its bounded result crosses
    // this channel: a late lookup never owns a socket or dispatches a request.
    std::thread::Builder::new()
        .name("vade-resolve".into())
        .spawn(move || {
            let _ = send.send(lookup());
        })?;
    receive
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .map_err(|error| match error {
            mpsc::RecvTimeoutError::Timeout => timeout(),
            mpsc::RecvTimeoutError::Disconnected => io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "HTTP hostname resolver stopped",
            ),
        })?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn stalled_resolution_does_not_hold_the_request_past_its_deadline() {
        let (release, stall) = mpsc::channel();
        let (started, entered) = mpsc::channel();
        let (finished, exited) = mpsc::channel();
        let began = Instant::now();
        let result = before(began + Duration::from_millis(50), move || {
            started.send(()).unwrap();
            let _ = stall.recv_timeout(Duration::from_secs(5));
            finished.send(()).unwrap();
            Ok(vec!["127.0.0.1:80".parse().unwrap()])
        });
        let elapsed = began.elapsed();
        // Release and observe the resolver before asserting, even on failure.
        release.send(()).unwrap();
        entered.recv_timeout(Duration::from_secs(1)).unwrap();
        exited.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::TimedOut);
        assert!(
            elapsed < Duration::from_secs(1),
            "resolver held request for {elapsed:?}"
        );
    }

    #[test]
    fn deadline_resolution_preserves_addresses_and_errors() {
        let peer: SocketAddr = "[::1]:8080".parse().unwrap();
        assert_eq!(
            before(Instant::now() + Duration::from_secs(1), move || Ok(vec![
                peer
            ]))
            .unwrap(),
            vec![peer]
        );
        assert_eq!(
            before(Instant::now() + Duration::from_secs(1), || Err(
                io::ErrorKind::NotFound.into()
            ))
            .unwrap_err()
            .kind(),
            io::ErrorKind::NotFound
        );
        assert_eq!(
            before(Instant::now(), || panic!(
                "expired request must not start resolution"
            ))
            .unwrap_err()
            .kind(),
            io::ErrorKind::TimedOut
        );
    }
}
