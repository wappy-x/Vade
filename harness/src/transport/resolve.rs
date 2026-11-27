//! A request deadline also bounds the wait for the operating system resolver.
use super::Endpoint;
use std::io;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::sync::{Condvar, Mutex, mpsc};
use std::time::Instant;

const MAX_RESOLVERS: usize = 8;
static RESOLVERS: ResolverSlots = ResolverSlots::new(MAX_RESOLVERS);

struct ResolverSlots {
    active: Mutex<usize>,
    available: Condvar,
    limit: usize,
}

struct ResolverSlot(&'static ResolverSlots);
impl Drop for ResolverSlot {
    fn drop(&mut self) {
        *self.0.active.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
        self.0.available.notify_one();
    }
}

impl ResolverSlots {
    const fn new(limit: usize) -> Self {
        Self { active: Mutex::new(0), available: Condvar::new(), limit }
    }
    fn acquire(&'static self, deadline: Instant) -> io::Result<ResolverSlot> {
        let mut active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() { return Err(timeout(true)); }
            if *active < self.limit {
                *active += 1;
                return Ok(ResolverSlot(self));
            }
            active = self.available.wait_timeout(active, left).unwrap_or_else(|e| e.into_inner()).0;
        }
    }
}

pub(crate) fn addresses(
    endpoint: &Endpoint,
    deadline: Instant,
) -> io::Result<Vec<SocketAddr>> {
    #[cfg(test)]
    if let Some(result) = TEST_LOOKUP.with(|hook| hook.borrow_mut().as_mut().map(|lookup| lookup(endpoint, deadline))) {
        return result;
    }
    if let Ok(ip) = endpoint.host.parse::<IpAddr>() {
        return Ok(vec![SocketAddr::new(ip, endpoint.port)]);
    }
    if endpoint.host.eq_ignore_ascii_case("localhost") {
        return Ok(vec![
            SocketAddr::new(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST), endpoint.port),
            SocketAddr::new(IpAddr::V6(std::net::Ipv6Addr::LOCALHOST), endpoint.port),
        ]);
    }
    let host = endpoint.host.clone();
    let port = endpoint.port;
    let lookup = move || {
        (host.as_str(), port)
            .to_socket_addrs()
            .map(|addresses| addresses.take(16).collect())
    };
    before(deadline, lookup).map_err(|error| {
        if error.kind() == io::ErrorKind::TimedOut {
            lookup_timeout(&endpoint.host, capacity_timeout(&error))
        } else { error }
    })
}

#[cfg(test)]
type TestLookup = Box<dyn FnMut(&Endpoint, Instant) -> io::Result<Vec<SocketAddr>>>;
#[cfg(test)]
thread_local! { static TEST_LOOKUP: std::cell::RefCell<Option<TestLookup>> = const { std::cell::RefCell::new(None) }; }
#[cfg(test)]
pub(crate) fn with_lookup<T>(lookup: TestLookup, run: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset { fn drop(&mut self) { TEST_LOOKUP.with(|hook| *hook.borrow_mut() = None); } }
    TEST_LOOKUP.with(|hook| *hook.borrow_mut() = Some(lookup));
    let _reset = Reset;
    run()
}

#[derive(Debug)]
pub(crate) struct LookupTimeout { pub host: String, waiting_for_helper: bool }
impl std::fmt::Display for LookupTimeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Name lookup for {} did not finish before the request deadline", self.host)?;
        if self.waiting_for_helper { f.write_str(" while waiting for a local resolver helper")?; }
        Ok(())
    }
}
impl std::error::Error for LookupTimeout {}

pub(crate) fn lookup_timeout(host: &str, waiting_for_helper: bool) -> io::Error {
    io::Error::new(io::ErrorKind::TimedOut, LookupTimeout { host: host.to_owned(), waiting_for_helper })
}
pub(crate) fn capacity_timeout(error: &io::Error) -> bool {
    error.get_ref().and_then(|e| e.downcast_ref::<LookupTimeout>()).is_some_and(|e| e.waiting_for_helper)
}
pub(crate) fn lookup_timed_out(error: &io::Error) -> bool {
    error.get_ref().is_some_and(|e| e.is::<LookupTimeout>())
}
fn timeout(waiting_for_helper: bool) -> io::Error {
    lookup_timeout("HTTP upstream", waiting_for_helper)
}

fn before(
    deadline: Instant,
    lookup: impl FnOnce() -> io::Result<Vec<SocketAddr>> + Send + 'static,
) -> io::Result<Vec<SocketAddr>> {
    before_with_slots(deadline, &RESOLVERS, lookup)
}

fn before_with_slots(
    deadline: Instant,
    slots: &'static ResolverSlots,
    lookup: impl FnOnce() -> io::Result<Vec<SocketAddr>> + Send + 'static,
) -> io::Result<Vec<SocketAddr>> {
    if Instant::now() >= deadline {
        return Err(timeout(false));
    }
    let slot = slots.acquire(deadline)?;
    let (send, receive) = mpsc::sync_channel(1);
    // The OS lookup itself cannot be cancelled. Only its bounded result crosses
    // this channel: a late lookup never owns a socket or dispatches a request.
    std::thread::Builder::new()
        .name("vade-resolve".into())
        .spawn(move || {
            let _slot = slot;
            let _ = send.send(lookup());
        })?;
    receive
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .map_err(|error| match error {
            mpsc::RecvTimeoutError::Timeout => timeout(false),
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

    #[test]
    fn stalled_helpers_stop_at_the_global_bound() {
        let slots = Box::leak(Box::new(ResolverSlots::new(2)));
        let (release, stalled) = mpsc::channel::<()>();
        let stalled = std::sync::Arc::new(std::sync::Mutex::new(stalled));
        for _ in 0..2 {
            let stalled = std::sync::Arc::clone(&stalled);
            let result = before_with_slots(Instant::now() + Duration::from_millis(30), slots, move || {
                let _ = stalled.lock().unwrap().recv_timeout(Duration::from_secs(2));
                Ok(vec![SocketAddr::new(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST), 1)])
            });
            assert_eq!(result.unwrap_err().kind(), io::ErrorKind::TimedOut);
        }
        let began = Instant::now();
        let refused = before_with_slots(Instant::now() + Duration::from_millis(60), slots, || panic!("no helper slot is free")).unwrap_err();
        assert_eq!(refused.kind(), io::ErrorKind::TimedOut);
        assert!(capacity_timeout(&refused));
        assert!(began.elapsed() >= Duration::from_millis(60));
        assert_eq!(*slots.active.lock().unwrap(), 2);
        release.send(()).unwrap();
        release.send(()).unwrap();
        let until = Instant::now() + Duration::from_secs(1);
        while *slots.active.lock().unwrap() != 0 && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(*slots.active.lock().unwrap(), 0);
    }

    #[test]
    fn sixty_four_localhost_lookups_need_no_helper() {
        let mut workers = Vec::new();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(64));
        // Even an expired helper deadline must not affect local addresses.
        for _ in 0..64 {
            let barrier = barrier.clone();
            workers.push(std::thread::spawn(move || {
                let endpoint = Endpoint::parse("http://localhost:80").unwrap();
                barrier.wait();
                addresses(&endpoint, Instant::now()).unwrap()
            }));
        }
        for worker in workers {
            assert_eq!(worker.join().unwrap(), vec!["127.0.0.1:80".parse().unwrap(), "[::1]:80".parse().unwrap()]);
        }
    }

    #[test]
    fn twelve_slow_name_lookups_wait_for_eight_helpers() {
        let slots = Box::leak(Box::new(ResolverSlots::new(MAX_RESOLVERS)));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(12));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut workers = Vec::new();
        for _ in 0..12 {
            let barrier = barrier.clone();
            let peak = peak.clone();
            let slots: &'static ResolverSlots = slots;
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                before_with_slots(Instant::now() + Duration::from_secs(2), slots, move || {
                    peak.fetch_max(*slots.active.lock().unwrap(), std::sync::atomic::Ordering::Relaxed);
                    std::thread::sleep(Duration::from_millis(60));
                    Ok(vec!["127.0.0.1:80".parse().unwrap()])
                })
            }));
        }
        for worker in workers { assert_eq!(worker.join().unwrap().unwrap().len(), 1); }
        assert!(peak.load(std::sync::atomic::Ordering::Relaxed) <= MAX_RESOLVERS);
    }
}
