//! Child grants and the shared parent/child approval lane.
use crate::{
    cli::{
        approval::ToolPreviews,
        args::{Config, Parsed},
        events::{WorkerEvent, WorkerResult},
    },
    interaction,
};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}, mpsc};
use std::time::{Duration, Instant};
use vade_harness::{coding_tools, shell::ApprovalPolicy};

fn config(extra: &[&str]) -> Result<Config, &'static str> {
    let mut args = vec![
        "agent",
        "--endpoint",
        "http://127.0.0.1:1",
        "--model",
        "mock",
        "--workspace",
        ".",
    ];
    args.extend(extra);
    match Config::parse(args.into_iter().map(str::to_owned)).map_err(|error| error.message)? {
        Parsed::Run(config) => Ok(config),
        _ => panic!(),
    }
}
fn fact(rx: &interaction::Inbox<WorkerEvent, WorkerResult>) -> WorkerEvent {
    let until = Instant::now() + Duration::from_secs(3);
    loop {
        match rx.fact() {
            Ok(event) => return event,
            Err(mpsc::TryRecvError::Empty) if Instant::now() < until => {
                std::thread::park_timeout(Duration::from_millis(1))
            }
            _ => panic!("approval fact did not arrive"),
        }
    }
}

#[test]
fn native_child_grants_require_host_opt_in_and_exact_bounded_paths() {
    assert!(config(&[]).unwrap().child_grants.is_empty());
    let edit_grants = ["--children", "--child-edit", "edits=src/a.rs", "--child-edit", "edits=src/b.rs"];
    let all_grants = [&edit_grants[..], &["--child-shell", "checks"]].concat();
    let shell = vade_harness::coding_tools::shell_offered();
    if !shell {
        // Linux, or VADE_HARNESS_SHELL=off: there is no shell to grant.
        assert!(config(&all_grants).err().unwrap().starts_with("--child-shell"));
    }
    let config = config(if shell { &all_grants } else { &edit_grants }).unwrap();
    assert_eq!(config.approval, ApprovalPolicy::Ask);
    let edits = &config.child_grants["edits"];
    assert_eq!(edits.write_paths, ["src/a.rs", "src/b.rs"]);
    assert!(edits.tools.iter().any(|name| name == "apply_patch"));
    assert!(!edits.tools.iter().any(|name| name == "workspace_shell"));
    if shell {
        let checks = &config.child_grants["checks"];
        assert!(checks.tools.iter().any(|name| name == "workspace_shell"));
        assert!(checks.write_paths.is_empty());
    }
    for args in [
        vec!["--child-edit", "edits=a"],
        vec!["--child-shell", "checks"],
        vec!["--children", "--child-edit", "edits"],
        vec!["--children", "--child-edit", "edits=../outside"],
        vec![
            "--children",
            "--child-edit",
            "edits=src/a",
            "--child-edit",
            "edits=src/A",
        ],
        vec![
            "--children",
            "--child-edit",
            "edits=src/a",
            "--child-edit",
            "edits=src",
        ],
        vec!["--children", "--child-edit", "edits=./a"],
        vec!["--children", "--child-shell", "bad scope"],
    ] {
        assert!(self::config(&args).is_err(), "{args:?}");
    }
}

#[test]
fn shared_parent_child_approval_lane_serializes_exact_operations_and_clears_receipts() {
    let (tx, rx) = interaction::lanes();
    let gate = Arc::new(ToolPreviews {
        tx: Arc::new(tx),
        bytes: Mutex::new(0),
        budget: None,
    });
    let first = Arc::clone(&gate);
    let a = std::thread::spawn(move || {
        first.show(
            None,
            coding_tools::Preview {
                title: "Patch preview",
                detail: "first exact operation",
                approval_required: true,
            },
            &AtomicBool::new(false),
        )
    });
    let WorkerEvent::ToolPreview {
        detail,
        reply: Some(reply),
        ..
    } = fact(&rx)
    else {
        panic!()
    };
    assert_eq!(detail, "first exact operation");
    let second = Arc::clone(&gate);
    let b = std::thread::spawn(move || {
        second.show(
            None,
            coding_tools::Preview {
                title: "Patch preview",
                detail: "second exact operation",
                approval_required: true,
            },
            &AtomicBool::new(false),
        )
    });
    assert!(
        matches!(rx.fact(), Err(mpsc::TryRecvError::Empty)),
        "first approval still owns the lane"
    );
    reply.send(true).unwrap();
    assert!(matches!(fact(&rx), WorkerEvent::ToolApprovalClosed));
    let WorkerEvent::ToolPreview {
        detail,
        reply: Some(reply),
        ..
    } = fact(&rx)
    else {
        panic!()
    };
    assert_eq!(detail, "second exact operation");
    assert!(a.join().unwrap());
    reply.send(false).unwrap();
    assert!(matches!(fact(&rx), WorkerEvent::ToolApprovalClosed));
    assert!(!b.join().unwrap());
}

#[test]
fn child_cancellation_interrupts_both_queued_and_active_approvals() {
    let (tx, rx) = interaction::lanes();
    let gate = Arc::new(ToolPreviews {
        tx: Arc::new(tx),
        bytes: Mutex::new(0),
        budget: None,
    });
    let active_cancel = Arc::new(AtomicBool::new(false));
    let queued_cancel = Arc::new(AtomicBool::new(false));
    let (done, results) = mpsc::channel();
    let spawn = |id, cancel: Arc<AtomicBool>| {
        let gate = Arc::clone(&gate);
        let done = done.clone();
        std::thread::spawn(move || {
            let ok = gate.show(
                None,
                coding_tools::Preview {
                    title: "Command",
                    detail: "exact command",
                    approval_required: true,
                },
                &cancel,
            );
            done.send((id, ok)).unwrap();
        })
    };
    let first = spawn(0, Arc::clone(&active_cancel));
    let WorkerEvent::ToolPreview {
        reply: Some(reply), ..
    } = fact(&rx)
    else {
        panic!()
    };
    let second = spawn(1, Arc::clone(&queued_cancel));
    queued_cancel.store(true, Ordering::Release);
    assert_eq!(
        results.recv_timeout(Duration::from_secs(2)).unwrap(),
        (1, false)
    );
    second.join().unwrap();
    assert!(
        matches!(rx.fact(), Err(mpsc::TryRecvError::Empty)),
        "queued cancelled approval was never shown"
    );
    active_cancel.store(true, Ordering::Release);
    assert_eq!(
        results.recv_timeout(Duration::from_secs(2)).unwrap(),
        (0, false)
    );
    first.join().unwrap();
    assert!(matches!(fact(&rx), WorkerEvent::ToolApprovalClosed));
    assert!(
        reply.send(true).is_err(),
        "stale approval cannot authorize an operation"
    );
}
