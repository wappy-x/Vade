//! Challenge-response admission for control and data peers. Subsequent traffic
//! is not encrypted; the shared secret never appears on these connections.
use crate::Result;
use std::{io::{Read, Write}, net::TcpStream, time::{Duration, Instant}};

// Leave room below the one-second rejection limit for socket teardown.
const TIMEOUT: Duration = Duration::from_millis(800);

pub fn control_secret() -> Result<Option<Vec<u8>>> {
    std::env::var_os("SEQPAR_CONTROL_SECRET").map(|value| {
        let bytes = value.into_encoded_bytes();
        if bytes.is_empty() { Err("SEQPAR_CONTROL_SECRET must not be empty".into()) }
        else { Ok(bytes) }
    }).transpose()
}

/// Accept and authenticate immediately while the outgoing neighbour starts.
/// No unauthenticated connection waits for another rank or reaches RDMA setup.
pub fn connect_peers(listen: Option<std::net::SocketAddr>, next: Option<std::net::SocketAddr>, end: Instant) -> Result<(Option<TcpStream>, Option<TcpStream>)> {
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
    let secret = control_secret()?;
    if secret.is_none() && listen.iter().chain(next.iter()).any(|address| !address.ip().is_loopback()) {
        return Err("non-loopback rank transport requires SEQPAR_CONTROL_SECRET".into());
    }
    let listener = listen.map(std::net::TcpListener::bind).transpose().map_err(|e| format!("listen: {e}"))?;
    if let Some(listener) = &listener { listener.set_nonblocking(true).map_err(|e| e.to_string())?; }
    let stopped = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let outgoing = scope.spawn(|| {
            let result = (|| {
                let Some(next) = next else { return Ok(None); };
                let mut stream = loop {
                    if stopped.load(Relaxed) { return Err("rank connection stopped".into()); }
                    let remaining = end.checked_duration_since(Instant::now()).ok_or("rank connection timed out")?;
                    match TcpStream::connect_timeout(&next, remaining.min(Duration::from_millis(100))) {
                        Ok(stream) => break stream,
                        Err(error) if Instant::now() >= end => return Err(format!("connect next rank: {error}")),
                        Err(_) => std::thread::sleep(Duration::from_millis(10)),
                    }
                };
                authenticate_peer(&mut stream, secret.as_deref(), 0, end.min(Instant::now() + TIMEOUT))?;
                Ok(Some(stream))
            })();
            if result.is_err() { stopped.store(true, Relaxed); }
            result
        });
        let incoming = listener.as_ref().map(|listener| {
            accept_peer(listener, secret.as_deref(), end, &stopped)
        }).transpose();
        if incoming.is_err() { stopped.store(true, Relaxed); }
        let outgoing = outgoing.join().map_err(|_| "rank authentication thread failed")?;
        Ok((outgoing?, incoming?))
    })
}

fn accept_peer(listener: &std::net::TcpListener, secret: Option<&[u8]>, end: Instant,
    stopped: &std::sync::atomic::AtomicBool) -> Result<TcpStream>
{
    use std::sync::atomic::Ordering::Relaxed;
    std::thread::scope(|scope| {
        let mut pending: Vec<(TcpStream, std::thread::ScopedJoinHandle<'_, Result<TcpStream>>)> = Vec::new();
        let result = (|| loop {
            if stopped.load(Relaxed) { return Err("rank connection stopped".into()); }
            if Instant::now() >= end { return Err("rank connection timed out".into()); }
            for index in (0..pending.len()).rev() {
                if pending[index].1.is_finished() {
                    let (_, task) = pending.swap_remove(index);
                    // A rejected connection never aborts the rank's admission window.
                    if let Ok(Ok(stream)) = task.join() { return Ok(stream); }
                }
            }
            // A silent client cannot occupy the only handshake slot. Bound both
            // concurrent work and accept rate, retaining the original deadline.
            if pending.len() < 4 {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let close = stream.try_clone().map_err(|e| e.to_string())?;
                        let deadline = end.min(Instant::now() + TIMEOUT);
                        let task = scope.spawn(move || {
                            authenticate_peer(&mut stream, secret, 1, deadline)?;
                            Ok(stream)
                        });
                        pending.push((close, task));
                    }
                    Err(error) if matches!(error.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted) => (),
                    Err(error) => return Err(format!("accept previous rank: {error}")),
                }
            }
            std::thread::sleep(Duration::from_millis(5));
        })();
        // Interrupt every losing handshake before joining, including silent peers.
        for (stream, _) in &pending { let _ = stream.shutdown(std::net::Shutdown::Both); }
        for (_, task) in pending { let _ = task.join(); }
        result
    })
}

// Client/server roles and both nonces bind the HMAC to this connection and
// prevent one peer from reflecting the other's proof back at it.
fn authenticate_peer(stream: &mut TcpStream, secret: Option<&[u8]>, role: u8, end: Instant) -> Result<()> {
    stream.set_nonblocking(false).map_err(|e| e.to_string())?;
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    let mut hello = [0u8; 42];
    hello[..8].copy_from_slice(b"SEQPEER1");
    hello[8] = u8::from(secret.is_some());
    hello[9] = role;
    if secret.is_some() {
        std::fs::File::open("/dev/urandom").and_then(|mut file| file.read_exact(&mut hello[10..]))
            .map_err(|_| "rank authentication entropy unavailable")?;
    }
    write(stream, &hello, end)?;
    let mut peer = [0u8; 42];
    read(stream, &mut peer, end)?;
    if peer[..9] != hello[..9] || peer[9] != 1 - role {
        return Err("rank authentication protocol/secret mismatch".into());
    }
    let Some(secret) = secret else { return Ok(()); };
    let mut transcript = [0u8; 85];
    let (client, server) = if role == 0 { (&hello, &peer) } else { (&peer, &hello) };
    transcript[..42].copy_from_slice(client);
    transcript[42..84].copy_from_slice(server);
    transcript[84] = role;
    write(stream, &hmac_sha256(secret, &transcript), end)?;
    transcript[84] = 1 - role;
    let expected = hmac_sha256(secret, &transcript);
    let mut received = [0u8; 32];
    read(stream, &mut received, end)?;
    if received.iter().zip(expected).fold(0u8, |n, (a, b)| n | (a ^ b)) != 0 {
        return Err("rank authentication rejected".into());
    }
    Ok(())
}

fn deadline(stream: &TcpStream, end: Instant) -> Result<()> {
    let remaining = end.checked_duration_since(Instant::now())
        .filter(|n| !n.is_zero()).ok_or("control authentication timed out")?;
    stream.set_read_timeout(Some(remaining)).map_err(|_| "control authentication socket error")?;
    stream.set_write_timeout(Some(remaining)).map_err(|_| "control authentication socket error".into())
}

// Reapply an absolute deadline after partial I/O so a trickling client
// cannot extend authentication one read at a time.
fn read(stream: &mut TcpStream, mut bytes: &mut [u8], end: Instant) -> Result<()> {
    while !bytes.is_empty() {
        deadline(stream, end)?;
        match stream.read(bytes) {
            Ok(0) => return Err("control authentication disconnected".into()),
            Ok(n) => bytes = &mut bytes[n..],
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err("control authentication read failed or timed out".into()),
        }
    }
    Ok(())
}

fn write(stream: &mut TcpStream, mut bytes: &[u8], end: Instant) -> Result<()> {
    while !bytes.is_empty() {
        deadline(stream, end)?;
        match stream.write(bytes) {
            Ok(0) => return Err("control authentication disconnected".into()),
            Ok(n) => bytes = &bytes[n..],
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err("control authentication write failed or timed out".into()),
        }
    }
    Ok(())
}

pub fn authenticate_coordinator(stream: &mut TcpStream, secret: Option<&[u8]>, end: Instant) -> Result<()> {
    authenticate_peer(stream, secret, 0, end.min(Instant::now() + TIMEOUT))
        .map_err(|error| format!("seqpar authentication failed: {error}"))
}

pub fn authenticate_worker(stream: &mut TcpStream, secret: Option<&[u8]>) -> Result<()> {
    authenticate_peer(stream, secret, 1, Instant::now() + TIMEOUT)
}

fn hmac_sha256(secret: &[u8], message: &[u8]) -> [u8; 32] {
    let mut key = [0u8; 64];
    if secret.len() > key.len() { key[..32].copy_from_slice(&sha256(secret)); }
    else { key[..secret.len()].copy_from_slice(secret); }
    let mut inner = vec![0u8; 64 + message.len()];
    let mut outer = [0u8; 96];
    for i in 0..64 { inner[i] = key[i] ^ 0x36; outer[i] = key[i] ^ 0x5c; }
    inner[64..].copy_from_slice(message);
    outer[64..].copy_from_slice(&sha256(&inner));
    sha256(&outer)
}

#[test]
fn hmac_sha256_rfc4231() {
    // RFC 4231 section 4.2, Test Case 1: https://www.rfc-editor.org/rfc/rfc4231#section-4.2
    assert_eq!(hmac_sha256(&[0x0b; 20], b"Hi There"), [
        0xb0, 0x34, 0x4c, 0x61, 0xd8, 0xdb, 0x38, 0x53, 0x5c, 0xa8, 0xaf, 0xce, 0xaf, 0x0b, 0xf1, 0x2b,
        0x88, 0x1d, 0xc2, 0x00, 0xc9, 0x83, 0x3d, 0xa7, 0x26, 0xe9, 0x37, 0x6c, 0x2e, 0x32, 0xcf, 0xf7]);
}
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn compress(state: &mut [u32; 8], block: &[u8]) {
    let mut w = [0u32; 64];
    for (word, bytes) in w.iter_mut().zip(block.chunks_exact(4)) {
        *word = u32::from_be_bytes(bytes.try_into().unwrap());
    }
    for i in 16..64 {
        let x = w[i - 15];
        let y = w[i - 2];
        let s0 = x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3);
        let s1 = y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choice = (e & f) ^ (!e & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(choice)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(majority);
        [h, g, f, e, d, c, b, a] = [g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2)];
    }
    for (word, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *word = word.wrapping_add(value);
    }
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut state = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut blocks = data.chunks_exact(64);
    for block in &mut blocks { compress(&mut state, block); }
    let rest = blocks.remainder();
    let mut tail = [0u8; 128];
    tail[..rest.len()].copy_from_slice(rest);
    tail[rest.len()] = 0x80;
    let padded = if rest.len() < 56 { 64 } else { 128 };
    tail[padded - 8..padded].copy_from_slice(&(data.len() as u64).wrapping_mul(8).to_be_bytes());
    for block in tail[..padded].chunks_exact(64) { compress(&mut state, block); }
    let mut result = [0u8; 32];
    for (bytes, word) in result.chunks_exact_mut(4).zip(state) { bytes.copy_from_slice(&word.to_be_bytes()); }
    result
}

/// A single deadline covers all peers, including peers that trickle replies.
pub const RELEASE_TIMEOUT: Duration = Duration::from_secs(2);
pub fn release_controls(controls: &mut [TcpStream], deadline: Instant) -> Result<()> {
    for control in controls.iter() { let _ = control.shutdown(std::net::Shutdown::Write); }
    let mut failure = None;
    let mut reply = [0u8; 256];
    for (peer, control) in controls.iter_mut().enumerate() {
        let drained = (|| {
            loop {
                let remaining = deadline.checked_duration_since(Instant::now())
                    .filter(|v| !v.is_zero()).ok_or_else(|| format!("seqpar peer {peer} release timed out"))?;
                control.set_read_timeout(Some(remaining)).map_err(|e| format!("seqpar peer {peer} release: {e}"))?;
                match control.read(&mut reply) {
                    Ok(0) => return Ok(()),
                    Ok(_) => (),
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => (),
                    Err(e) => return Err(format!("seqpar peer {peer} release: {e}")),
                }
            }
        })();
        let _ = control.shutdown(std::net::Shutdown::Both);
        if let Err(error) = drained { failure.get_or_insert(error); }
    }
    failure.map_or(Ok(()), Err)
}
#[cfg(test)]
#[test]
fn release_has_a_deadline() {
    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (_silent, _) = listener.accept().unwrap();
    assert!(release_controls(&mut [client], Instant::now() + Duration::from_millis(5)).unwrap_err().contains("peer 0 release"));
}
