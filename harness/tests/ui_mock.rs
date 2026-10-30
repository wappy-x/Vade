//! Actual CLI/raw-terminal lifecycle checks against local scripted sockets only.
#![cfg(target_os = "macos")]
#[path = "support/home.rs"]
mod home;
#[path = "support/cli.rs"]
mod cli;
#[path = "support/raw_ui.rs"]
mod raw_ui;
use raw_ui::Ui;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::process::Stdio;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

fn request(reader: &mut BufReader<std::net::TcpStream>) -> Vec<u8> {
    let mut size = None;
    let mut discovery = false;
    loop {
        let mut line = String::new();
        assert!(reader.read_line(&mut line).unwrap() > 0);
        discovery |= line == "GET /v1/capabilities HTTP/1.1\r\n";
        if line == "\r\n" {
            break;
        }
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            size = Some(value.trim().parse::<usize>().unwrap());
        }
    }
    let mut body = vec![0; size.unwrap()];
    reader.read_exact(&mut body).unwrap();
    if discovery {
        assert!(body.is_empty());
        reader
            .get_mut()
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
        return request(reader);
    }
    body
}

#[test]
fn unnamed_tty_launches_stay_interactive_with_inherited_nonblocking_flags() {
    let path = std::env::temp_dir().join(format!("vade-fresh-ui-{}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    for args in [
        &["--session", "saved"][..],
        &[],
        &[],
        &["--resume", "saved"],
    ] {
        let mut ui = Ui::start_nonblocking(path.to_str().unwrap(), args);
        thread::sleep(Duration::from_millis(100)); // idle input used to abort the UI
        ui.send(b"/");
        let menu = ui.until("Esc close", Duration::from_secs(2));
        assert!(!menu.contains("Last reply"));
        ui.send(b"\x15/quit\r");
        let receipt = ui.until("resume with --resume", Duration::from_secs(2));
        if !args.is_empty() {
            assert!(receipt.contains("Session: saved"));
        }
        ui.wait_exit();
    }
    let workspace = vade_harness::tools::Workspace::new(&path).unwrap();
    assert_eq!(vade_harness::journal::list(&workspace).unwrap().len(), 3);
    drop(workspace);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn raw_status_ticks_redraw_one_row_without_repeated_newlines() {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut reader = BufReader::new(socket);
        request(&mut reader);
        assert_eq!(reader.read(&mut [0]).unwrap(), 0);
    });
    let mut ui = Ui::start("agent", &endpoint, ".", false);
    ui.send(b"wait\r");
    ui.until("thinking…", Duration::from_secs(2));
    let mut ticks = String::new();
    for (rows, columns) in [(12, 24), (30, 120), (8, 18), (24, 80)] {
        ui.resize(rows, columns);
        ticks.push_str(&ui.until("thinking…", Duration::from_secs(2)));
    }
    assert!(ticks.contains("\x1b[1B\r"));
    assert!(
        !ticks.contains('\n'),
        "status ticks must stay out of scrollback: {ticks:?}"
    );
    assert!(ticks.matches("thinking…").count() >= 4);
    ui.send(&[27]);
    server.join().unwrap();
}

#[test]
fn raw_exit_keeps_session_and_reply_tail_and_wheel_leaves_draft_unchanged() {
    let workspace = std::env::temp_dir().join(format!("vade-exit-ui-{}", std::process::id()));
    std::fs::create_dir(&workspace).unwrap();
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(socket);
        request(&mut reader);
        let reply = format!(
            "{}tail one\ntail two 🦀\ntail three",
            "history line\n".repeat(50)
        );
        let body =
            vade_harness::json::stringify(&vade_harness::json::Value::String(reply)).unwrap();
        write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {{\"choices\":[{{\"delta\":{{\"content\":{body}}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n").unwrap();
    });
    let mut ui = Ui::start_with_args(
        "agent",
        &endpoint,
        workspace.to_str().unwrap(),
        &["--session", "exit-receipt"],
    );
    ui.send(b"old prompt\r");
    let answer = ui.until("╰ ", Duration::from_secs(3));
    for hidden in [
        "Native run:",
        "Run speed:",
        "contract:",
        "provisional output",
        "model response 1:",
    ] {
        assert!(!answer.contains(hidden), "default UI leaked {hidden}");
    }
    ui.send(b"current draft\x1b[<64;12;5M");
    let history = ui.until("history · PgDn", Duration::from_secs(2));
    assert!(history.contains("current draft"));
    assert!(!history.contains("old prompt"));
    ui.send(b"\x1b[<65;12;5M\x15/quit\r");
    let exited = ui.until(
        "Last reply (tail):\r\ntail one\r\ntail two 🦀\r\ntail three",
        Duration::from_secs(2),
    );
    let receipt = exited
        .split("\x1b[?1049l")
        .nth(1)
        .expect("normal screen restored");
    assert!(receipt.contains("Session: exit-receipt · resume with --resume exit-receipt"));
    assert!(receipt.contains("Last reply (tail):\r\ntail one\r\ntail two 🦀\r\ntail three"));
    assert!(!receipt.contains("history line"));
    server.join().unwrap();
    drop(ui);
    std::fs::remove_dir_all(workspace).unwrap();
}

#[test]
fn plain_tty_status_prints_only_once_while_the_state_stays_queued() {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut reader = BufReader::new(socket);
        request(&mut reader);
        assert_eq!(reader.read(&mut [0]).unwrap(), 0);
    });
    let mut ui = Ui::start_with_args("agent", &endpoint, ".", &["--plain", "--verbose"]);
    ui.send(b"wait\r");
    ui.until("queued / awaiting upstream", Duration::from_secs(2));
    thread::sleep(Duration::from_millis(450));
    ui.send(b"/cancel\r");
    let ticks = ui.until("Native run:", Duration::from_secs(2));
    assert!(
        !ticks.contains("queued / awaiting upstream"),
        "plain mode repeated its state: {ticks:?}"
    );
    server.join().unwrap();
}

#[test]
fn plain_tty_streamed_reply_is_uninterrupted_and_state_resets_for_the_next_run() {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let (continue_tx, continue_rx) = mpsc::channel();
    let server = thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut reader = BufReader::new(socket);
        request(&mut reader);
        reader.get_mut().write_all("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hidden reasoning\",\"content\":\"\\n\\n\"}}]}\n\n".as_bytes()).unwrap();
        continue_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        reader
            .get_mut()
            .write_all("data: {\"choices\":[{\"delta\":{\"content\":\"\\nHel\"}}]}\n\n".as_bytes())
            .unwrap();
        continue_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        reader.get_mut().write_all("data: {\"choices\":[{\"delta\":{\"content\":\"lo 🦀\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".as_bytes()).unwrap();
        drop(reader);
        let (socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut reader = BufReader::new(socket);
        request(&mut reader);
        assert_eq!(reader.read(&mut [0]).unwrap(), 0);
    });
    let mut ui = Ui::start_with_args(
        "agent",
        &endpoint,
        ".",
        &["--plain", "--verbose", "--thinking", "on"],
    );
    ui.send(b"stream\r");
    let mut transcript = ui.until("thinking", Duration::from_secs(2));
    continue_tx.send(()).unwrap();
    transcript.push_str(&ui.until("Hel", Duration::from_secs(2)));
    // Allow the Generating state (sent after the first native delta) to redraw
    // while the model's next delta is deliberately held back.
    thread::sleep(Duration::from_millis(250));
    continue_tx.send(()).unwrap();
    transcript.push_str(&ui.until("Native run:", Duration::from_secs(2)));
    assert!(
        transcript.contains("Hello 🦀\r\n"),
        "plain streaming was interrupted or its final line was not closed: {transcript:?}"
    );
    ui.send(b"again\r");
    ui.until("queued / awaiting upstream", Duration::from_secs(2));
    ui.send(b"/cancel\r");
    ui.until("Native run:", Duration::from_secs(2));
    server.join().unwrap();
}

#[test]
fn raw_ui_cancels_while_mock_has_sent_no_response_bytes() {
    for mode in ["direct", "agent"] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (ready_tx, ready_rx) = mpsc::channel();
        let server = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut reader = BufReader::new(socket);
            request(&mut reader);
            ready_tx.send(()).unwrap();
            // The peer sends nothing. Cancellation must close this socket.
            assert_eq!(reader.read(&mut [0]).unwrap(), 0);
        });
        let mut ui = Ui::start(mode, &endpoint, ".", false);
        ui.send(b"stall\r");
        ready_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        // Controls remain live while no response bytes exist, but mutable
        // history/client state stays owned by the in-flight request.
        ui.send(b"/new\r");
        ui.until(
            "Wait for request cleanup before /new.",
            Duration::from_secs(1),
        );
        ui.send(b"/undo\r");
        ui.until(
            "Wait for active request cleanup before /undo.",
            Duration::from_secs(1),
        );
        ui.send(b"/context\r");
        ui.until("the current one reports when it finishes.", Duration::from_secs(1));
        let start = Instant::now();
        ui.send(&[27]);
        let text = ui.until("cancellation requested.", Duration::from_secs(1));
        assert!(text.contains("cancellation requested."));
        assert!(start.elapsed() < Duration::from_secs(1));
        // Native/direct worker cleanup polls the socket independently; the UI
        // cancellation notice arrives without any upstream response/EOF.
        server.join().unwrap();
        ui.send(b"/quit\r");
    }
}

#[test]
fn cli_reuses_native_and_direct_clients_across_user_prompts() {
    for mode in ["direct", "agent"] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut reader = BufReader::new(socket);
            let mut previous = Vec::new();
            for turn in 0..3 {
                let body = request(&mut reader);
                assert!(body.starts_with(&previous));
                let value = vade_harness::json::parse_bytes(&body).unwrap();
                for message in value
                    .get("messages")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|m| {
                        m.get("role").and_then(vade_harness::json::Value::as_str)
                            == Some("assistant")
                    })
                {
                    assert_eq!(
                        message.get("content").unwrap().as_str(),
                        Some("\n\n\n  \nanswer")
                    );
                }
                previous = body[..body.len() - 2].to_vec();
                let response = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hidden reasoning\",\"content\":\"\\n\\n\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"\\n  \\nanswer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
                let response = if turn == 0 {
                    "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"reasoning_content\":\"hidden reasoning\",\"content\":\"\\n\\n\\n  \\nanswer\"},\"finish_reason\":\"stop\"}]}"
                } else {
                    response
                };
                let kind = if turn == 0 {
                    "application/json"
                } else {
                    "text/event-stream"
                };
                write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n\r\n{response}", response.len()).unwrap();
            }
        });
        let mut child = home::harness()
            .args([
                mode,
                "--state-in-project",
                "--endpoint",
                &endpoint,
                "--model",
                "mock",
                "--workspace",
                ".",
                "--plain",
                "--thinking",
                "on",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(b"/queue on\nfirst\nsecond\nthird\n")
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success());
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            "Queueing on: typed messages wait for the current reply.\nanswer\nanswer\nanswer\n"
        );
        server.join().unwrap();
    }
}

/// Inside /paste, a piped /exit is message text, not an exit (10-30 train
/// review).
#[test]
fn piped_exit_inside_paste_is_text() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut reader = BufReader::new(socket);
        let body = vade_harness::json::parse_bytes(&request(&mut reader)).unwrap();
        let messages = body.get("messages").unwrap().as_array().unwrap();
        assert_eq!(messages.last().unwrap().get("content").unwrap().as_str(), Some("keep\n/exit"));
        let response = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}";
        write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{response}", response.len()).unwrap();
    });
    let mut child = home::harness()
        .args(["direct", "--state-in-project", "--endpoint", &endpoint, "--model", "mock", "--plain"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"/paste\nkeep\n/exit\n/end\n").unwrap();
    let output = child.wait_with_output().unwrap();
    let (stdout, stderr) = (String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "{stdout}{stderr}");
    assert!(stdout.ends_with("\nanswer\n"), "{stdout}{stderr}");
    server.join().unwrap();
}

#[test]
fn direct_budget_requires_explicit_new_before_prefix_rewrite() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut reader = BufReader::new(socket);
        let mut prefix = Vec::new();
        for turn in 0..34 {
            let body = request(&mut reader);
            if turn < 33 {
                assert!(body.starts_with(&prefix));
                prefix = body[..body.len() - 2].to_vec();
            } else {
                let value = vade_harness::json::parse_bytes(&body).unwrap();
                let messages = value.get("messages").unwrap().as_array().unwrap();
                assert_eq!(
                    messages.len(),
                    2,
                    "only explicit /new may reset this prefix"
                );
                assert_eq!(messages[1].get("content").unwrap().as_str(), Some("fresh"));
            }
            let response = "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
            write!(reader.get_mut(), "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{response}", response.len()).unwrap();
        }
    });
    let mut command = home::harness();
    command.args([
        "direct",
        "--state-in-project",
        "--endpoint",
        &endpoint,
        "--model",
        "mock",
        "--workspace",
        ".",
        "--plain",
    ]);
    let mut child = cli::Cli::start(&mut command);
    for _ in 0..33 {
        child.send("prompt\n");
        child.until(true, " · ctx ");
    }
    child.send("prompt\n");
    child.until(
        false,
        "nothing was sent. Use /new before the next message",
    );
    child.send("/new\nfresh\n");
    let output = child.finish();
    assert!(
        !output.status.success(),
        "the blocked 34th prompt must report failure"
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text.lines().filter(|line| *line == "answer").count(), 34);
    assert!(text.contains("nothing was sent. Use /new before the next message"));
    server.join().unwrap();
}

#[test]
fn raw_ui_shows_tool_call_rows_and_result_summaries_without_verbose() {
    let workspace = std::env::temp_dir().join(format!("vade-tool-rows-{}", std::process::id()));
    std::fs::create_dir(&workspace).unwrap();
    std::fs::write(workspace.join("input.txt"), "\r\nsource @untouched\r\n").unwrap();
    std::fs::write(workspace.join("binary"), b"\0").unwrap();
    let expected = "make the file @input.txt\n\n[attached files]\n[file: input.txt, 2 lines]\n\r\nsource @untouched\r\n\n";
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        let replies = [
            r#"{"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":"I'll create the file.","tool_calls":[{"id":"edit-1","type":"function","function":{"name":"apply_patch","arguments":"{\"patch\":\"*** Begin Patch\\n*** Add File: result.txt\\n+exact result\\n*** End Patch\"}"}}]}}]}"#,
            r#"{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"done"}}]}"#,
        ];
        for reply in replies {
            let (socket, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(socket);
            let body = request(&mut reader);
            assert!(vade_harness::json::parse(std::str::from_utf8(&body).unwrap()).unwrap().get("messages").unwrap().as_array().unwrap().iter().any(|m| m.get("content").and_then(vade_harness::json::Value::as_str) == Some(expected)));
            write!(
                reader.get_mut(),
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                reply.len(),
                reply
            )
            .unwrap();
        }
    });
    let mut ui = Ui::start_with_args(
        "agent",
        &endpoint,
        workspace.to_str().unwrap(),
        &["--approval", "auto", "--session", "attachments"],
    );
    ui.send(b"keep this @binary\r"); let refused = ui.until("Draft kept in the editor.", Duration::from_secs(2)); assert!(refused.contains("keep this @binary"));
    ui.send(b"\x15/paste\r@binary\r/end\r"); ui.until("Draft kept in the editor.", Duration::from_secs(2));
    ui.send(b"\x15make the file @inp\t\r");
    let shown = ui.until("done", Duration::from_secs(5));
    server.join().unwrap();
    for expected in [
        "[file: input.txt, 2 lines]",
        "● Edit workspace files",
        "Patch preview",
        "+exact result",
        "⎿ file written · ",
        "result.txt",
        "⎿ applied 1 file operations · ",
        "done",
    ] {
        assert!(
            shown.contains(expected),
            "missing {expected:?} in {shown:?}"
        );
    }
    for hidden in ["Patch result", "Applied 1 file operations.", "Native run:"] {
        assert!(!shown.contains(hidden), "default UI leaked {hidden}");
    }
    assert!(shown.find("I'll create the file.").unwrap() < shown.find("● Edit workspace files").unwrap());
    assert_eq!(
        std::fs::read_to_string(workspace.join("result.txt")).unwrap(),
        "exact result\n"
    );
    ui.send(b"/quit\r");
    drop(ui);
    let history = std::fs::read(workspace.join(".vade/history")).unwrap(); assert!(!String::from_utf8_lossy(&history).contains("source @untouched") && !String::from_utf8_lossy(&history).contains("[attached files]"));
    std::fs::write(workspace.join("input.txt"), "changed").unwrap();
    let root = vade_harness::tools::Workspace::new(&workspace).unwrap();
    let (_, resumed, _) = vade_harness::journal::Journal::resume(&root, "attachments", vade_harness::agent::Limits::default()).unwrap();
    assert!(resumed.messages().iter().any(|m| m.value().get("content").and_then(vade_harness::json::Value::as_str) == Some(expected))); drop(root);
    std::fs::remove_dir_all(workspace).unwrap();
}
