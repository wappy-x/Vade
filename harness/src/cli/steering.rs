//! Preserve interrupted prose separately from closed tool receipts.
use super::{App, events::Pending, native_setup::NativeResult};
use std::io;
use vade_harness::agent::Limits;
impl App {
    pub(crate) fn retain_native_interruption(&mut self, pending: &Pending, native: &mut NativeResult, finished: bool) -> io::Result<()> {
        if pending.canceled && !finished {
            let partial = pending.provisional.get(pending.render_skip..).unwrap_or("");
            let mut conversation = native.conversation.clone();
            let recovery = match conversation.push_interrupted_answer(partial, Limits::default()) {
                Ok(truncated) => {
                    self.journal.as_ref().unwrap().lock().map_err(|_| io::Error::other("journal lock failed"))?.save(&conversation)?;
                    native.conversation = conversation;
                    truncated.then_some("Conversation full: the interrupted reply was shortened in saved history; its interruption marker was kept.")
                }
                Err(error) if error.kind() == io::ErrorKind::InvalidData => {
                    Some("Conversation full: no room to save the interrupted reply or its marker; the earlier history is unchanged.")
                }
                Err(error) => return Err(error),
            };
            self.terminal.answer(&format!("{partial}\n[interrupted]"))?;
            if let Some(reason) = recovery {
                if let Some(text) = self.steering.take() { self.queue(text)?; }
                self.failed = true;
                self.run_notice(&format!("{reason} Follow-up held; use /new, then /next (or recall the draft with Up)."))?;
            }
        }
        // A generation that demonstrably finished needs no cancellation
        // acknowledgment; a late refusal must not hold the follow-up.
        if !finished && native.model.server_cancel_confirmed() == Some(false) {
            if let Some(text) = self.steering.take() { self.queue(text)?; }
            self.failed = true;
            self.run_notice("Server cancellation was not acknowledged; follow-up held for /next. Check the server first.")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::args;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{mpsc, Arc, atomic::{AtomicBool, Ordering}};
    use std::time::{Duration, Instant};
    use vade_harness::json;

    fn reply(socket: &mut TcpStream, body: &str) {
        write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    }

    fn drive(app: &mut super::App, what: &str, done: impl Fn(&super::App) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !done(app) {
            app.poll_worker().unwrap();
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::park_timeout(Duration::from_millis(5));
        }
        app.poll_worker().unwrap();
    }

    fn mock_server() -> (String, mpsc::Receiver<(TcpStream, json::Value)>, mpsc::Receiver<String>, Arc<AtomicBool>, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (requests_tx, requests) = mpsc::channel::<(TcpStream, json::Value)>();
        let (cancels_tx, cancels) = mpsc::channel::<String>();
        let stop = Arc::new(AtomicBool::new(false));
        let server_stop = Arc::clone(&stop);
        let server = std::thread::spawn(move || loop {
            if server_stop.load(Ordering::Relaxed) { break; }
            let (socket, _) = match listener.accept() {
                Ok(socket) => socket,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(1)); continue;
                }
                Err(error) => panic!("{error}"),
            };
            socket.set_nonblocking(false).unwrap();
            socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut reader = BufReader::new(socket);
            let mut first = String::new(); reader.read_line(&mut first).unwrap();
            let mut size = 0;
            loop {
                let mut line = String::new(); assert!(reader.read_line(&mut line).unwrap() > 0);
                if line == "\r\n" { break; }
                if let Some(n) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    size = n.trim().parse().unwrap();
                }
            }
            let mut bytes = vec![0; size]; reader.read_exact(&mut bytes).unwrap();
            let mut socket = reader.into_inner();
            if first.starts_with("GET /v1/capabilities ") {
                let caps = include_str!("../../tests/fixtures/engine-capabilities-v1.json")
                    .replace("\"/v1/admission\"", "\"unavailable\"").replace("4096", "262144").replace("4000", "262000");
                reply(&mut socket, &caps);
            } else if first.starts_with("POST /v1/cancel ") {
                let body = json::parse_bytes(&bytes).unwrap();
                cancels_tx.send(body.get("cancel").unwrap().as_str().unwrap().to_owned()).unwrap();
                reply(&mut socket, r#"{"cancelled":true}"#);
            } else {
                assert!(first.starts_with("POST /v1/chat/completions "), "{first}");
                requests_tx.send((socket, json::parse_bytes(&bytes).unwrap())).unwrap();
            }
        });
        (endpoint, requests, cancels, stop, server)
    }

    fn texts(body: &json::Value) -> Vec<(String, String)> {
        body.get("messages").unwrap().as_array().unwrap().iter().map(|message| (
            message.get("role").unwrap().as_str().unwrap().to_owned(),
            message.get("content").and_then(json::Value::as_str).unwrap_or_default().to_owned(),
        )).collect()
    }

    fn held_pending() -> (super::super::events::WorkSender, super::Pending) {
        let (tx, rx) = crate::interaction::lanes();
        let (controls, _) = mpsc::sync_channel(8);
        (tx, super::Pending {
            budget: None, native_progress: None, user: "original".into(),
            started: Instant::now(), cancel: Arc::new(AtomicBool::new(false)),
            rx, controls, steering_pending: Vec::new(), render_skipped: false,
            render_skip: 0, round_displayed: false, received: false,
            canceled: false, displayed: 0, provisional: String::new(),
            edits: 0, worker: None, join_on_drop: false,
        })
    }

    #[test]
    fn successive_steering_over_aggregate_limit_keeps_rejected_input() {
        use vade_harness::terminal::Key;
        let args::Parsed::Run(config) = args::Config::parse(
            ["direct", "--endpoint", "http://127.0.0.1:1", "--model", "mock", "--plain"]
                .into_iter().map(str::to_owned),
        ).unwrap() else { panic!("expected direct configuration") };
        let mut app = super::App::new(config).unwrap();
        // Keep the producer alive without a terminal result. Cancellation
        // cannot finish while both real editor submissions are consumed.
        let (_producer, pending) = held_pending();
        app.pending = Some(pending);
        let first = "P".repeat(40 * 1024);
        let second = "Q".repeat(30 * 1024);
        app.input.set(first.clone());
        app.key(Key::Enter).unwrap();
        assert!(app.pending.as_ref().unwrap().canceled);
        app.input.set(second.clone());
        app.key(Key::Enter).unwrap();
        assert_eq!(app.steering.as_deref(), Some(first.as_str()));
        assert!(app.pending.is_some(), "cleanup is still held open");
        app.key(Key::Up).unwrap();
        assert!(app.input.text() == second, "rejected steering must be recoverable from the draft");
    }

    #[test]
    fn steer_end_line_says_the_new_message_follows() {
        use vade_harness::agent::{RunReport, StopReason};
        let root = std::env::temp_dir().join(format!("vade-steer-end-line-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir(&root).unwrap();
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
        let _cleanup = Cleanup(root.clone());
        let args::Parsed::Run(config) = args::Config::parse(
            ["agent", "--endpoint", "http://127.0.0.1:1", "--model", "mock", "--workspace", root.to_str().unwrap(), "--plain"]
                .into_iter().map(str::to_owned),
        ).unwrap() else { panic!("expected native configuration") };
        let mut app = super::App::new(config).unwrap();
        let report = RunReport {
            reason: StopReason::Cancelled, answer_cut_off: false, model_turns: 1, auto_continues: 0, nudges: 0,
            auto_checks: 0, tool_turns: 0, tool_calls: 0, output_bytes: 0,
        };
        let (_producer, mut pending) = held_pending();
        pending.canceled = true;
        let line = |app: &super::App, pending: &super::Pending| {
            let line = app.run_end_line(&report, false, pending, false, false);
            let parts: Vec<String> = line.split(" · ").map(str::to_owned).collect();
            (parts[0].clone(), parts[2..].join(" · "))
        };
        // Esc: the reply was canceled, and continue resumes it.
        assert_eq!(line(&app, &pending), ("Canceled".into(), "no edits · say continue".into()));
        // A steer: the reply stopped for the new message, which goes next.
        app.steering = Some("correction".into());
        assert_eq!(line(&app, &pending), ("Stopped for your new message".into(), "no edits · sending it now".into()));
        // A quit after the steer (Ctrl+C on an empty editor, SIGINT) sends nothing.
        app.quit = true;
        assert_eq!(line(&app, &pending), ("Canceled".into(), "no edits · say continue".into()));
    }

    fn native_capacity_case(room: usize) {
        use super::super::{events::{TurnResult, WorkerResult}, native_setup::native_message};
        use vade_harness::agent::{Limits, RunReport, StopReason};
        let root = std::env::temp_dir().join(format!("vade-native-steer-capacity-{}-{room}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
        let _cleanup = Cleanup(root.clone());
        let args::Parsed::Run(config) = args::Config::parse(
            ["agent", "--endpoint", "http://127.0.0.1:1", "--model", "mock", "--workspace",
                root.to_str().unwrap(), "--session", "capacity", "--plain"]
                .into_iter().map(str::to_owned),
        ).unwrap() else { panic!("expected native configuration") };
        let mut app = super::App::new(config).unwrap();
        let limits = Limits::default();
        let mut conversation = app.native.take().unwrap();
        let mut empty = conversation.clone();
        empty.push_user(native_message("user", ""), limits).unwrap();
        let padding = "x".repeat(limits.max_history_bytes - empty.serialized_bytes() - room);
        conversation.push_user(native_message("user", &padding), limits).unwrap();
        assert_eq!(conversation.serialized_bytes(), limits.max_history_bytes - room);
        app.journal.as_ref().unwrap().lock().unwrap().save(&conversation).unwrap();
        let before = conversation.messages().to_vec();
        let native = super::NativeResult {
            model: vade_harness::transport::NativeModel::new("http://127.0.0.1:1").unwrap().with_server_cancellation(), conversation,
            report: RunReport {
                reason: StopReason::Cancelled, answer_cut_off: false,
                model_turns: 1, auto_continues: 0, nudges: 0, auto_checks: 0,
                tool_turns: 0, tool_calls: 0, output_bytes: 0,
            },
            check_failing: false, check_not_run: false, display_limited: false,
            children: Vec::new(), answer: String::new(),
        };
        let (producer, mut pending) = held_pending();
        pending.canceled = true;
        let partial = "\"\\\né🙂".repeat(100);
        pending.provisional = partial.clone();
        app.pending = Some(pending);
        app.steering = Some("held correction".into());
        producer.finish(WorkerResult { http: None, result: Ok(TurnResult::Native(native)), backend: None });
        assert!(app.poll_worker().is_ok(), "history capacity must not escape the UI event loop");
        assert!(app.pending.is_none());
        assert!(app.native_model.is_some(), "restore native client after capacity refusal");
        let restored = app.native.as_ref().expect("restore authoritative conversation");
        assert_eq!(&restored.messages()[..before.len()], before);
        assert!(restored.serialized_bytes() <= limits.max_history_bytes);
        let retained = if room > 0 {
            let text = restored.messages().last().unwrap().value().get("content").unwrap().as_str().unwrap();
            let prefix = text.strip_suffix("\n[interrupted]").expect("reserve the interruption marker");
            assert!(!prefix.is_empty() && prefix.len() < partial.len());
            assert!(partial.starts_with(prefix), "truncate on a UTF-8 boundary without changing prose");
            Some(text.to_owned())
        } else {
            assert_eq!(restored.messages(), before);
            None
        };
        assert!(app.steering.is_none(), "do not auto-dispatch into a full conversation");
        assert_eq!(app.queued.pop().as_deref(), Some("held correction"));
        assert!(!app.quit, "capacity is a recoverable UI condition");
        drop(app);
        let workspace = vade_harness::tools::Workspace::new(&root).unwrap();
        let sessions = vade_harness::journal::list(&workspace).unwrap();
        let (_store, reopened, _torn) = vade_harness::journal::Journal::resume(&workspace, &sessions[0], limits).unwrap();
        assert_eq!(&reopened.messages()[..before.len()], before);
        if let Some(retained) = retained {
            assert_eq!(reopened.messages().last().unwrap().value().get("content").unwrap().as_str(), Some(retained.as_str()));
        }
    }

    #[test]
    fn native_interrupted_history_near_capacity_reserves_marker() {
        native_capacity_case(128);
    }

    #[test]
    fn native_interrupted_history_without_marker_room_holds_correction() {
        native_capacity_case(0);
    }

    /// Portable native steering coverage without a PTY: drive the App event
    /// loop pieces directly against the contract mock, then reopen the
    /// journal to prove the interrupted partial answer was persisted.
    #[test]
    fn native_steering_retains_partial_and_journal() {
        let root = std::env::temp_dir().join(format!("vade-native-steer-unit-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
        let _cleanup = Cleanup(root.clone());
        let (endpoint, requests, cancels, stop, server) = mock_server();
        let args::Parsed::Run(config) = args::Config::parse(
            ["agent", "--endpoint", &endpoint, "--model", "mock", "--workspace",
                root.to_str().unwrap(), "--session", "steer", "--plain"]
                .into_iter().map(str::to_owned),
        ).unwrap() else { panic!("expected native configuration") };
        let mut app = super::App::new(config).unwrap();
        app.line_input("original".into()).unwrap();
        // Generation starts only after the admission count is polled through.
        let (mut first, body) = {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                app.poll_worker().unwrap();
                if let Ok(request) = requests.try_recv() { break request; }
                assert!(Instant::now() < deadline, "generation never started");
                std::thread::park_timeout(Duration::from_millis(5));
            }
        };
        first.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").unwrap();
        write!(first, "data: {{\"choices\":[{{\"delta\":{{\"content\":\"PART_UNIT\"}}}}]}}\n\n").unwrap();
        drive(&mut app, "provisional text", |app| {
            app.pending.as_ref().is_some_and(|pending| pending.provisional.contains("PART_UNIT"))
        });
        app.line_input("replacement".into()).unwrap();
        assert_eq!(cancels.recv_timeout(Duration::from_secs(5)).unwrap(),
            body.get("cancel").unwrap().as_str().unwrap());
        drive(&mut app, "interrupted cleanup", |app| app.pending.is_none());
        assert!(!app.failed);
        // The run loop dispatches steering once cleanup finishes; mirror it.
        let steering = app.steering.take().expect("steering text held for dispatch");
        app.start(steering).unwrap();
        let (mut next, body) = {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                app.poll_worker().unwrap();
                if let Ok(request) = requests.try_recv() { break request; }
                assert!(Instant::now() < deadline, "steering generation never started");
                std::thread::park_timeout(Duration::from_millis(5));
            }
        };
        let history = texts(&body);
        assert!(history.windows(3).any(|messages| messages == [
            ("user".to_owned(), "original".to_owned()),
            ("assistant".to_owned(), "PART_UNIT\n[interrupted]".to_owned()),
            ("user".to_owned(), "replacement".to_owned()),
        ]), "history: {history:?}");
        reply(&mut next, r#"{"choices":[{"message":{"role":"assistant","content":"FINAL"},"finish_reason":"stop"}]}"#);
        drop(next);
        drive(&mut app, "final cleanup", |app| app.pending.is_none());
        assert!(!app.failed);
        drop(app);
        let workspace = vade_harness::tools::Workspace::new(&root).unwrap();
        let sessions = vade_harness::journal::list(&workspace).unwrap();
        assert_eq!(sessions.len(), 1);
        let (_store, conversation, _torn) = vade_harness::journal::Journal::resume(
            &workspace, &sessions[0], vade_harness::agent::Limits::default()).unwrap();
        let contents: Vec<String> = conversation.messages().iter()
            .filter(|message| message.role() == "assistant")
            .filter_map(|message| message.value().get("content").and_then(json::Value::as_str).map(str::to_owned))
            .collect();
        assert!(contents.iter().any(|text| text == "PART_UNIT\n[interrupted]"),
            "reopened journal assistant messages: {contents:?}");
        stop.store(true, Ordering::Relaxed);
        server.join().unwrap();
    }

    /// Deterministic end-of-stream race: the generation worker has fully
    /// finished (validated EOF) but the UI has not polled its result yet
    /// when the steering message is entered. The completed answer must be
    /// recorded unmarked and the message must become the next turn.
    #[test]
    fn direct_message_after_worker_finished_is_a_normal_next_turn() {
        let (endpoint, requests, _cancels, stop, server) = mock_server();
        let args::Parsed::Run(config) = args::Config::parse(
            ["direct", "--endpoint", &endpoint, "--model", "mock", "--plain"]
                .into_iter().map(str::to_owned),
        ).unwrap() else { panic!("expected direct configuration") };
        let mut app = super::App::new(config).unwrap();
        app.line_input("original".into()).unwrap();
        let (mut first, _body) = requests.recv_timeout(Duration::from_secs(5)).unwrap();
        first.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").unwrap();
        first.write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"FULL_UNIT\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n").unwrap();
        drop(first);
        // Wait for the worker thread itself, without polling its result.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !app.pending.as_ref().and_then(|pending| pending.worker.as_ref())
            .is_some_and(std::thread::JoinHandle::is_finished)
        {
            assert!(Instant::now() < deadline, "generation worker never finished");
            std::thread::park_timeout(Duration::from_millis(5));
        }
        app.line_input("at the end".into()).unwrap();
        assert!(app.steering.is_some(), "typed message held as steering");
        drive(&mut app, "completion cleanup", |app| app.pending.is_none());
        let history: Vec<(&str, &str)> = app.session.history().iter()
            .map(|message| (message.role.as_str(), message.content.as_str())).collect();
        assert_eq!(history, [("user", "original"), ("assistant", "FULL_UNIT")],
            "finished answer recorded unmarked");
        assert!(!app.failed);
        let steering = app.steering.take().expect("steering text held for dispatch");
        app.start(steering).unwrap();
        let (mut next, body) = requests.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(&texts(&body)[1..], [
            ("user".to_owned(), "original".to_owned()),
            ("assistant".to_owned(), "FULL_UNIT".to_owned()),
            ("user".to_owned(), "at the end".to_owned()),
        ]);
        reply(&mut next, r#"{"choices":[{"message":{"role":"assistant","content":"FINAL"},"finish_reason":"stop"}]}"#);
        drop(next);
        drive(&mut app, "final cleanup", |app| app.pending.is_none());
        assert!(!app.failed);
        drop(app);
        stop.store(true, Ordering::Relaxed);
        server.join().unwrap();
    }
}
