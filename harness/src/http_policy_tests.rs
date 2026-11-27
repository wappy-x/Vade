use super::*;
use crate::transport::{self, ConnectionPool};
use std::{io::{Read, Write}, net::TcpListener, thread, time::Duration};

#[test]
fn connected_address_rule() {
    for (peer, local, allow, expected) in [
        ("127.0.0.1:80", "192.0.2.20:1", false, true),
        ("[::1]:80", "[2001:db8::2]:1", false, true),
        ("[::ffff:127.0.0.1]:80", "[2001:db8::2]:1", false, true),
        ("192.0.2.10:80", "192.0.2.10:1", false, true),
        ("[2001:db8::1]:80", "[2001:db8::1]:1", false, true),
        ("[::ffff:192.0.2.10]:80", "192.0.2.10:1", false, true),
        ("192.0.2.10:80", "192.0.2.20:1", false, false),
        ("[2001:db8::1]:80", "[2001:db8::2]:1", false, false),
        ("192.0.2.10:80", "192.0.2.20:1", true, true),
    ] { assert_eq!(allowed(peer.parse().unwrap(), local.parse().unwrap(), allow), expected, "{peer} {local}"); }
}
#[test]
fn refusal_preserves_typed_host_and_explains_tunnel() {
    let endpoint = Endpoint::parse("http://Model.Example:018080/v1").unwrap();
    let message = full_reason(&endpoint);
    for text in ["Model.Example:018080", "Model.Example is another machine", "plain HTTP", "not encrypted", "read and change", "ssh -N -L", "--allow-plain-http", "018080:127.0.0.1:018080", "use http://127.0.0.1:018080 as the address"] {
        assert!(message.contains(text), "{message}");
    }
}

/// Real loopback wire, fake connected addresses. A forbidden send must close
/// its socket; the listener sees EOF with no HTTP bytes, including headers.
pub(crate) fn wire(allow: bool, send: impl FnOnce(&str) -> io::Result<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        let mut bytes = Vec::new();
        let mut b = [0; 1];
        while socket.read(&mut b).unwrap() != 0 {
            bytes.push(b[0]);
            if bytes.ends_with(b"\r\n\r\n") {
                let header = String::from_utf8_lossy(&bytes);
                let length = header.lines().find_map(|line| line.strip_prefix("Content-Length: ")).and_then(|n| n.trim().parse::<usize>().ok()).unwrap_or(0);
                let mut body = vec![0; length]; socket.read_exact(&mut body).unwrap(); bytes.extend(body);
                let _ = socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                break;
            }
        }
        bytes
    });
    let (result, logs) = testing::remote(allow, || send(&url));
    let bytes = server.join().unwrap();
    if allow { assert!(result.is_ok(), "{result:?}"); assert!(!bytes.is_empty()); assert!(logs.is_empty()); }
    else { assert!(result.is_err()); assert!(bytes.is_empty(), "{bytes:?}"); assert_eq!(logs.len(), 1); }
}
#[test]
fn pooled_request_refuses_before_writing_and_switch_allows() {
    for allow in [false, true] {
        wire(allow, |url| {
            let endpoint = Endpoint::parse(url)?;
            transport::request_method_pooled("GET", &endpoint, &[], None, None, &ConnectionPool::default()).map(drop)
        });
    }
}
#[test]
fn fresh_connection_refuses_before_writing_and_switch_allows() {
    for allow in [false, true] {
        wire(allow, |url| {
            let endpoint = Endpoint::parse(url)?;
            let mut stream = transport::connect_fresh(&endpoint, std::time::Instant::now() + Duration::from_secs(5))?;
            stream.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        });
    }
}
#[test]
fn test_override_is_restored_after_unwind() {
    let _ = std::panic::catch_unwind(|| testing::remote(false, || panic!("fixture")));
    assert!(testing::addresses().is_none());
}

#[test]
fn fatal_refusal_process() {
    // A refusal must unwind normally now. The real process failure status and
    // helper cleanup are also exercised by tests/http_policy_programs.rs.
    let dropped = std::cell::Cell::new(false);
    struct Guard<'a>(&'a std::cell::Cell<bool>);
    impl Drop for Guard<'_> { fn drop(&mut self) { self.0.set(true); } }
    wire(false, |url| {
        let _guard = Guard(&dropped);
        let endpoint = Endpoint::parse(url)?;
        transport::request_method_pooled("GET", &endpoint, &[], None, None, &ConnectionPool::default()).map(drop)
    });
    assert!(dropped.get(), "refusal skipped the caller's destructor");
}

#[test]
fn report_state_logs_once_and_clears_current_refusal_after_success() {
    let endpoint = Endpoint::parse("http://model.example:80").unwrap();
    let (_, lines) = testing::remote(false, || {
        for _ in 0..3 { with_reports(|reports| reports.note(&endpoint, true)); }
        assert!(last_refusal(&endpoint).unwrap().contains("--allow-plain-http"));
        with_reports(|reports| reports.note(&endpoint, false));
        assert!(last_refusal(&endpoint).is_none());
        with_reports(|reports| reports.note(&endpoint, true));
    });
    assert_eq!(lines, vec![full_reason(&endpoint)]);
}

#[test]
fn tunnel_hint_names_no_option_and_avoids_a_local_port_that_needs_administrator_rights() {
    // No port typed means port 80. A local port under 1024 needs administrator rights, so the hint uses 8080.
    let message = full_reason(&Endpoint::parse("http://model.example/v1").unwrap());
    assert!(message.contains("ssh -N -L 8080:127.0.0.1:80 USER@model.example"), "{message}");
    assert!(message.contains("use http://127.0.0.1:8080 as the address"), "{message}");
    // The doctor tool takes the address as its first word, so the text names no option.
    assert!(!message.contains("--endpoint") && !message.contains("--upstream"), "{message}");
}

#[test]
fn quality_keeps_typed_authority_for_refusals() {
    let endpoint = crate::quality::parse_endpoint("http://Model.Example:018080/v1").unwrap();
    assert_eq!(endpoint.authority, "Model.Example:018080");
    let endpoint = crate::quality::parse_endpoint("http://[::1]:018080/v1").unwrap();
    assert_eq!(endpoint.authority, "[::1]:018080");
}
