use super::{check_requests, compare_cached, tests::{reply, wire}};
use std::{io::{Read, Write}, net::{TcpListener, TcpStream}, time::Duration};

// Request order: returned run.sh on source 0e1dafc, 2026-11-03.
// run.out supplies the group order and successful four-token outputs.
// run.err supplies the final branch's reused=128 and shared=8.
// One-token and long replies below are synthetic. The returned records do
// not contain those IDs. This fixture proves checker behavior, not GPU IDs.
fn request(socket: &mut TcpStream) -> String {
    socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let mut bytes = Vec::new();
    let mut buffer = [0; 4096];
    loop {
        let n = socket.read(&mut buffer).unwrap(); assert_ne!(n, 0);
        bytes.extend_from_slice(&buffer[..n]);
        if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            let header = std::str::from_utf8(&bytes[..end]).unwrap();
            let length = header.lines().find_map(|line| line.strip_prefix("Content-Length: "))
                .map(|s| s.parse::<usize>().unwrap()).unwrap_or(0);
            if bytes.len() == end + 4 + length { return String::from_utf8(bytes).unwrap(); }
        }
    }
}

#[test]
fn returned_plain_then_strict_order_accepts_a_prior_divergent_prefix() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let mut history: Vec<Vec<u32>> = Vec::new();
        let mut branches = Vec::new();
        // The returned tool stopped during the last group. Continue that
        // group, then repeat both probe modes to check a warm server too.
        for (length, draft) in [(16, "plain"), (16, "strict"), (128, "plain"), (128, "strict"), (128, "plain"), (128, "strict")] {
            let prompt = [1, 42, 1024, 17].repeat(length / 4);
            let mut divergent = prompt.clone(); divergent[8] = 220;
            let short = prompt[..4].to_vec();
            let background = [1, 42, 1024, 17].repeat(128);
            let steps = [
                (&prompt, 1, false, false), (&prompt, 1, false, true),
                (&prompt, 4, false, false), (&prompt, 4, false, true),
                (&divergent, 4, false, true), (&divergent, 4, true, false),
                (&prompt, 4, false, true), (&short, 4, false, false),
                (&background, 128, false, false), (&short, 4, false, false),
                (&short, 64, false, false),
            ];
            for (index, (expected, output, disabled, session)) in steps.into_iter().enumerate() {
                let (mut socket, _) = listener.accept().unwrap();
                let request = request(&mut socket);
                let body = request.split_once("\r\n\r\n").unwrap().1;
                let value = engine_model::json::parse(body.as_bytes(), Default::default()).unwrap();
                let object = value.object().unwrap();
                let ids: Vec<_> = object["prompt_token_ids"].array().unwrap().iter()
                    .map(|v| v.u64().unwrap() as u32).collect();
                assert_eq!(ids, *expected, "{length} {draft} step={index}");
                assert_eq!(object["max_tokens"].u64().unwrap(), output);
                assert_eq!(object["temperature"].u64().unwrap(), 0);
                assert_eq!(object["draft"].string().unwrap(), draft);
                assert_eq!(body.contains("\"cache\":false"), disabled);
                assert_eq!(object.contains_key("session"), session);
                assert_eq!(body.contains("\"fork\":true"), session);
                if index == 8 {
                    // The client drops this socket after the prefill marker.
                    socket.write_all(b"HTTP/1.1 200 OK\r\n\r\n: prefill\n\n").unwrap();
                    continue;
                }
                // Model only the aligned snapshots seen in the return. Use
                // every prior cache-enabled request, independent of draft.
                let shared = history.iter().map(|cached| cached.iter().zip(&ids)
                    .take_while(|(a, b)| a == b).count()).max().unwrap_or(0);
                let reused = if disabled { 0 } else { shared / 64 * 64 };
                if index == 4 { branches.push((length, draft, reused, shared)); }
                if !disabled && ids.len() >= 64 && !history.contains(&ids) { history.push(ids); }
                let tokens = if output == 1 { vec![42] } else { [1, 42, 1024, 17].repeat(output as usize / 4) };
                let response = wire(&tokens, reused, "").replace("test-key", &format!("key-{length}-{draft}-{index}"));
                socket.write_all(response.as_bytes()).unwrap();
            }
            let (mut socket, _) = listener.accept().unwrap();
            assert!(request(&mut socket).starts_with("GET /health "));
            socket.write_all(b"HTTP/1.1 200 OK\r\n\r\n{\"active\":0,\"queued\":0}").unwrap();
        }
        assert_eq!(branches, vec![(16, "plain", 0, 0), (16, "strict", 0, 0),
            (128, "plain", 0, 8), (128, "strict", 128, 128),
            (128, "plain", 128, 128), (128, "strict", 128, 128)]);
    });
    for (length, draft) in [(16, "plain"), (16, "strict"), (128, "plain"), (128, "strict"), (128, "plain"), (128, "strict")] {
        check_requests(address, 64, &[1, 42, 1024, 17].repeat(length / 4), Some(draft), false).unwrap();
    }
    server.join().unwrap();
}

#[test]
fn a_full_automatic_hit_keeps_token_and_accounting_failures() {
    let reference = reply(&[1, 42, 1024, 17], 0);
    assert_eq!(compare_cached("branch", &reference, &reply(&reference.tokens, 128), 128).unwrap(), "PASS");
    assert!(compare_cached("branch", &reference, &reply(&reference.tokens, 129), 128)
        .unwrap_err().contains("invalid reuse count"));
    for reused in [0, 64, 128] {
        for tokens in [&[1, 43, 1024, 17][..], &[1, 42, 1024][..], &[1, 42, 1024, 17, 5][..]] {
            assert!(compare_cached("branch", &reference, &reply(tokens, reused), 128)
                .unwrap_err().contains("token-ID mismatch"));
        }
    }
}
