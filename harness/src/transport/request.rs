//! Pooled HTTP requests: the engine, native and gateway entry points over one policy.
use super::*;

pub(crate) fn request_pooled<'a>(
    endpoint: &Endpoint,
    body: &[u8],
    forwarded: Option<&Head>,
    cancel: Option<&'a AtomicBool>,
    pool: &ConnectionPool,
) -> io::Result<Response<'a>> {
    request_method_pooled("POST", endpoint, body, forwarded, cancel, pool)
}

pub(crate) fn request_method_pooled<'a>(
    method: &str,
    endpoint: &Endpoint,
    body: &[u8],
    forwarded: Option<&Head>,
    cancel: Option<&'a AtomicBool>,
    pool: &ConnectionPool,
) -> io::Result<Response<'a>> {
    request_method_pooled_deadline(method, endpoint, body, forwarded, cancel, pool, None)
}

/// Optional tighter absolute budget, including waiting for DNS resolution.
/// No request is retried after dispatch.
pub(crate) fn request_method_pooled_deadline<'a>(
    method: &str,
    endpoint: &Endpoint,
    body: &[u8],
    forwarded: Option<&Head>,
    cancel: Option<&'a AtomicBool>,
    pool: &ConnectionPool,
    deadline: Option<Instant>,
) -> io::Result<Response<'a>> {
    request_with_policy(
        method,
        endpoint,
        body,
        forwarded,
        cancel,
        pool,
        RequestPolicy {
            deadline,
            downstream: None,
            response_idle_timeout: IDLE_TIMEOUT,
            on_dispatched: None,
            on_connect_failed: None,
        },
    )
}

/// Generation can wait behind queued work or a long prefill, even after HTTP
/// headers or role-only SSE events arrive. Retain the connection until the
/// absolute 3,600-second lifetime or owner cancellation. Uploads, discovery,
/// admission and direct chat retain their ordinary idle bound.
pub(crate) fn request_native_pooled<'a>(
    endpoint: &Endpoint,
    body: &[u8],
    cancel: &'a AtomicBool,
    pool: &ConnectionPool,
    on_dispatched: Option<&dyn Fn()>,
) -> io::Result<Response<'a>> {
    request_native_pooled_deadline(endpoint, body, cancel, pool, on_dispatched, None)
}

pub(crate) fn request_native_pooled_deadline<'a>(
    endpoint: &Endpoint,
    body: &[u8],
    cancel: &'a AtomicBool,
    pool: &ConnectionPool,
    on_dispatched: Option<&dyn Fn()>,
    deadline: Option<Instant>,
) -> io::Result<Response<'a>> {
    request_with_policy(
        "POST",
        endpoint,
        body,
        None,
        Some(cancel),
        pool,
        RequestPolicy {
            deadline,
            downstream: None,
            response_idle_timeout: LIFETIME,
            on_dispatched,
            on_connect_failed: None,
        },
    )
}

pub(super) struct RequestPolicy<'a> {
    pub(super) deadline: Option<Instant>,
    pub(super) downstream: Option<Arc<Mutex<Downstream>>>,
    pub(super) response_idle_timeout: Duration,
    pub(super) on_dispatched: Option<&'a dyn Fn()>,
    pub(super) on_connect_failed: Option<&'a dyn Fn()>,
}

pub(super) fn connect_deadline(deadline: Option<Instant>, now: Instant) -> Instant {
    deadline.map_or(now + CONNECT_TIMEOUT, |at| at.min(now + CONNECT_TIMEOUT))
}

/// Open a checked socket. Lookup and connect share one deadline, and the
/// plain-HTTP rule runs on the new socket before any caller can write to it.
pub(crate) fn connect_fresh(endpoint: &Endpoint, deadline: Instant) -> io::Result<TcpStream> {
    crate::http_policy::interactive_result()?;
    let deadline = connect_deadline(Some(deadline), Instant::now());
    connect_with(endpoint, deadline, resolve::addresses, TcpStream::connect_timeout)
}

fn connect_with(
    endpoint: &Endpoint,
    deadline: Instant,
    lookup: impl FnOnce(&Endpoint, Instant) -> io::Result<Vec<std::net::SocketAddr>>,
    mut connect: impl FnMut(&std::net::SocketAddr, Duration) -> io::Result<TcpStream>,
) -> io::Result<TcpStream> {
    let addresses = lookup(endpoint, deadline)?;
    for address in addresses {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() { break; }
        if let Ok(stream) = connect(&address, remaining) {
            return crate::http_policy::check(stream, endpoint);
        }
    }
    if Instant::now() >= deadline {
        Err(io::Error::new(io::ErrorKind::TimedOut, "HTTP request deadline exceeded"))
    } else {
        Err(io::Error::new(io::ErrorKind::ConnectionRefused, "unable to connect to HTTP upstream"))
    }
}

#[cfg(test)]
mod connect_tests {
    use super::*;

    #[test]
    fn lookup_and_connect_spend_one_shared_deadline() {
        let endpoint = Endpoint::parse("http://lookup.invalid:80").unwrap();
        let started = Instant::now();
        let deadline = started + Duration::from_millis(160);
        let mut tried = false;
        let result = connect_with(&endpoint, deadline, |_, lookup_deadline| {
            assert_eq!(lookup_deadline, deadline);
            std::thread::sleep(Duration::from_millis(80));
            Ok(vec!["127.0.0.1:80".parse().unwrap()])
        }, |_, left| {
            tried = true;
            assert!(left <= Duration::from_millis(80), "connect reset the lookup budget: {left:?}");
            std::thread::sleep(left);
            Err(io::ErrorKind::TimedOut.into())
        });
        assert!(tried);
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn real_request_limits_lookup_to_five_seconds() {
        let endpoint = Endpoint::parse("http://lookup.invalid:80").unwrap();
        resolve::with_lookup(Box::new(|_, deadline| {
            let left = deadline.saturating_duration_since(Instant::now());
            assert!(left <= Duration::from_secs(5) && left > Duration::from_secs(4), "lookup budget: {left:?}");
            Err(resolve::lookup_timeout("lookup.invalid", false))
        }), || {
            let result = request_pooled(&endpoint, b"{}", None, None, &ConnectionPool::default());
            assert_eq!(result.err().unwrap().kind(), io::ErrorKind::TimedOut);
        });
    }
}

pub(crate) struct GatewayFailure {
    /// DNS/connect failed on a fresh socket, before any request bytes existed.
    /// Write/read errors and stale pooled connections are always uncertain.
    pub connect_failed: bool,
    pub error: io::Error,
}

pub(crate) fn request_gateway_pooled<'a>(
    method: &str,
    endpoint: &Endpoint,
    body: &[u8],
    forwarded: &Head,
    cancel: Option<&'a AtomicBool>,
    pool: &ConnectionPool,
    downstream: &DeadlineStream<'_>,
) -> Result<Response<'a>, GatewayFailure> {
    let connect_failed = std::cell::Cell::new(false);
    let peer = downstream
        .buffered_input
        .clone()
        .ok_or_else(|| GatewayFailure {
            connect_failed: false,
            error: invalid("downstream monitoring requires a complete request"),
        })?;
    request_with_policy(
        method,
        endpoint,
        body,
        Some(forwarded),
        cancel,
        pool,
        RequestPolicy {
            deadline: Some(downstream.request_deadline()),
            downstream: Some(peer),
            response_idle_timeout: gateway_response_idle(method, &endpoint.path),
            on_dispatched: None,
            on_connect_failed: Some(&|| connect_failed.set(true)),
        },
    )
    .map_err(|error| GatewayFailure {
        connect_failed: connect_failed.get(),
        error,
    })
}

/// One gateway control call (cancel or checkpoint release) sent to several
/// endpoints at once. Every call shares the client's disconnect monitoring and
/// one absolute deadline, and the value can move to another thread.
#[derive(Clone)]
pub(crate) struct ControlFanout {
    peer: Arc<Mutex<Downstream>>,
    deadline: Instant,
}

impl ControlFanout {
    /// Calls end by `budget` from now, or the client request's own deadline.
    pub(crate) fn new(downstream: &DeadlineStream<'_>, budget: Duration) -> io::Result<Self> {
        let peer = downstream
            .buffered_input
            .clone()
            .ok_or_else(|| invalid("downstream monitoring requires a complete request"))?;
        let deadline = downstream.request_deadline().min(Instant::now() + budget);
        Ok(Self { peer, deadline })
    }

    /// Send the call to one endpoint. Reading the reply body is bounded by the
    /// same deadline; nothing is retried.
    pub(crate) fn request<'a>(
        &self,
        method: &str,
        endpoint: &Endpoint,
        body: &[u8],
        forwarded: &Head,
        cancel: Option<&'a AtomicBool>,
        pool: &ConnectionPool,
    ) -> io::Result<Response<'a>> {
        request_with_policy(
            method,
            endpoint,
            body,
            Some(forwarded),
            cancel,
            pool,
            RequestPolicy {
                deadline: Some(self.deadline),
                downstream: Some(Arc::clone(&self.peer)),
                response_idle_timeout: IDLE_TIMEOUT,
                on_dispatched: None,
                on_connect_failed: None,
            },
        )
    }
}

pub(super) fn gateway_response_idle(method: &str, path: &str) -> Duration {
    let path = path.split('?').next().unwrap_or(path);
    // Long cold prefills and admission queues need the full request lifetime,
    // including non-streaming responses that produce no bytes until complete.
    if method == "POST"
        && [
            "/messages",
            "/chat/completions",
            "/responses",
            "/completions",
        ]
        .iter()
        .any(|suffix| path.ends_with(suffix))
    {
        LIFETIME
    } else {
        IDLE_TIMEOUT
    }
}

pub(super) fn request_with_policy<'a>(
    method: &str,
    endpoint: &Endpoint,
    body: &[u8],
    forwarded: Option<&Head>,
    cancel: Option<&'a AtomicBool>,
    pool: &ConnectionPool,
    policy: RequestPolicy<'_>,
) -> io::Result<Response<'a>> {
    crate::http_policy::interactive_result()?;
    let deadline = policy.deadline;
    if !matches!(method, "GET" | "POST") || (method == "GET" && !body.is_empty()) {
        return Err(invalid("unsupported HTTP method or GET body"));
    }
    // The gateway has already bounded client input at MAX_INPUT_BYTES. Allow
    // its inserted defaults to grow a full-size request without lowering that
    // client limit; the existing transport ceiling still bounds wire bodies.
    let body_limit = if policy.downstream.is_some() { MAX_BODY } else { json::MAX_INPUT_BYTES };
    if body.len() > body_limit {
        return Err(invalid("HTTP request body exceeds limit"));
    }
    if endpoint.path.len() > MAX_LINE {
        return Err(invalid("HTTP endpoint path exceeds limit"));
    }
    if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        return Err(io::Error::new(
            io::ErrorKind::ConnectionAborted,
            "request canceled",
        ));
    }
    if deadline.is_some_and(|at| Instant::now() >= at) {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "HTTP request deadline exceeded",
        ));
    }
    if let Some(downstream) = &policy.downstream {
        downstream
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .check()?;
    }
    let api_key = crate::auth::ApiKey::configured()?;
    let socket = if let Some(socket) = pool.take(endpoint) {
        socket
    } else {
        let attempt = connect_fresh(endpoint, connect_deadline(deadline, Instant::now()));
        match attempt {
            Ok(socket) => socket,
            Err(error) => {
                if !resolve::capacity_timeout(&error)
                    && crate::http_policy::refusal(&error).is_none()
                    && let Some(on_failed) = policy.on_connect_failed {
                    on_failed();
                }
                return Err(error);
            }
        }
    };
    let mut socket = DeadlineStream::new(socket, cancel)?;
    socket.deadline = deadline;
    let relay = policy.downstream.is_some();
    socket.downstream = policy.downstream;
    let mut headers = format!(
        "{method} {} HTTP/1.1\r\nHost: {}\r\nContent-Length: {}\r\nConnection: keep-alive\r\n",
        endpoint.path,
        endpoint.authority,
        body.len()
    );
    if let Some(head) = forwarded {
        for (name, value) in &head.fields {
            if !hop_header(name, head)
                && !(api_key.is_some() && matches!(name.as_str(), "authorization" | "x-api-key"))
            {
                headers.push_str(name);
                headers.push_str(": ");
                headers.push_str(value);
                headers.push_str("\r\n");
            }
        }
    } else {
        headers.push_str("Content-Type: application/json\r\nAccept: text/event-stream, application/json\r\nAccept-Encoding: identity\r\n");
    }
    if let Some(key) = &api_key {
        headers.push_str(&key.bearer_header());
    }
    if headers.len() + 2 > MAX_HEADERS {
        return Err(invalid("forwarded HTTP headers exceed limit"));
    }
    headers.push_str("\r\n");
    socket.write_all(headers.as_bytes())?;
    socket.write_all(body)?;
    socket.flush()?;
    socket.idle_timeout = policy.response_idle_timeout;
    if let Some(on_dispatched) = policy.on_dispatched {
        on_dispatched();
    }
    let mut reader = BufReader::with_capacity(READ_BUFFER_BYTES, socket);
    let mut informational = 0;
    let (head, status) = loop {
        let head = read_head(&mut reader)?;
        let mut parts = head.first.splitn(3, ' ');
        if !matches!(parts.next(), Some("HTTP/1.1" | "HTTP/1.0")) {
            return Err(invalid("unsupported HTTP response version"));
        }
        let code = parts.next().ok_or_else(|| invalid("missing HTTP status"))?;
        if code.len() != 3 || !code.bytes().all(|b| b.is_ascii_digit()) {
            return Err(invalid("invalid HTTP status"));
        }
        let status = code
            .parse::<u16>()
            .map_err(|_| invalid("invalid HTTP status"))?;
        if !(100..=599).contains(&status) {
            return Err(invalid("invalid HTTP status"));
        }
        if (100..200).contains(&status) {
            informational += 1;
            if status == 101 || informational > 8 {
                return Err(invalid(
                    "unsupported HTTP protocol upgrade or informational response count",
                ));
            }
            continue;
        }
        break (head, status);
    };
    if status == 401 && !relay {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "HTTP 401 Unauthorized: authentication failed",
        ));
    }
    // A 304 may describe a compressed, arbitrarily large representation while
    // carrying no response body. Its metadata does not enable body decoding.
    let body = if matches!(status, 204 | 304) {
        BodyReader::empty(reader)
    } else {
        BodyReader::new(reader, &head, false, MAX_BODY)?
    };
    let reuse = if head.first.starts_with("HTTP/1.1 ") && !head.connection_close() {
        Some((pool.clone(), endpoint.authority.clone()))
    } else {
        None
    };
    Ok(Response {
        status,
        head,
        body,
        reuse,
    })
}
