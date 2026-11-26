//! Worker cleanup and bounded result summaries.
use super::*;

#[test]
fn pending_drop_disconnects_all_lanes_before_joining() {
    let (tx, rx) = interaction::lanes();
    let (controls, _) = mpsc::sync_channel(8);
    let finished = Arc::new(AtomicBool::new(false));
    let worker_finished = Arc::clone(&finished);
    let worker = std::thread::spawn(move || {
        for _ in 0..10000 {
            tx.render("fixture output");
        }
        tx.finish(WorkerResult {
            http: None,
            result: Ok(TurnResult::NotStarted),
            backend: None,
        });
        worker_finished.store(true, Ordering::Relaxed);
    });
    let pending = Pending {
        budget: None,
        native_progress: None,
        user: String::new(),
        started: Instant::now(),
        cancel: Arc::new(AtomicBool::new(false)),
        rx,
        controls,
        steering_pending: Vec::new(),
        render_skipped: false,
        render_skip: 0,
        round_displayed: false,
        received: false,
        canceled: false,
        displayed: 0,
        provisional: String::new(),
        edits: 0,
        worker: Some(worker),
        join_on_drop: true,
    };
    drop(pending);
    assert!(finished.load(Ordering::Relaxed));
}

#[test]
fn error_categories_do_not_leak_sensitive_details() {
    let error = io::Error::other("sensitive URL and response content");
    assert_eq!(error_category(&error), "transport error");
    assert!(error_hint(&error).contains("vade status"));
    assert!(!error_hint(&error).contains("sensitive"));
    let refused = io::Error::from(io::ErrorKind::ConnectionRefused);
    assert!(error_hint(&refused).contains("vade status"));
}

#[test]
fn tool_result_rows_summarize_command_and_patch_receipts() {
    let ok = "status: Exited(Some(0))\nstdout:\nline one\n\nline two\nstderr:\n";
    assert_eq!(
        command_summary(ok),
        ("exit 0 · 2 lines".into(), false, None)
    );
    let failed = "status: Exited(Some(101))\nstdout:\n\nstderr:\nerror[E0425]: cannot find value\n  --> src/main.rs:1:1\n";
    assert_eq!(
        command_summary(failed),
        (
            "exit 101 · 2 stderr".into(),
            true,
            Some("error[E0425]: cannot find value".into())
        )
    );
    for detail in [
        "status: Timeout\nstdout:\n\nstderr:\n",
        "status: Timeout\nstdout:\nstderr:\n",
    ] {
        assert_eq!(command_summary(detail), ("timeout".into(), true, None));
    }
    assert_eq!(
        patch_summary("Applied 2 file operations."),
        ("applied 2 file operations".into(), false, None)
    );
    assert!(patch_summary("{\"ok\":false}").1);
    assert_eq!(
        patch_summary(r#"{"ok":false,"uncertain_writes":false}"#),
        (
            "patch refused · no files written · request a new preview".into(),
            true,
            None
        )
    );
    assert!(
        patch_summary(r#"{"ok":false,"uncertain_writes":true}"#)
            .0
            .contains("uncertain")
    );
}
