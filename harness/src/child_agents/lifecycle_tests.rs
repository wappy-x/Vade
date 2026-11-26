//! Child lifecycle: bounded concurrency, joined workers, cancellation from the
//! parent or the UI, and lifecycle wakeups without polling.
use super::*;

#[test]
fn concurrency_is_bounded_and_drop_joins_every_owned_worker() {
    let (tx, rx) = mpsc::sync_channel(4);
    let live = Arc::new(AtomicUsize::new(0));
    let mut scheduler = scheduler(
        Config {
            max_concurrent: 2,
            ..Config::default()
        },
        Arc::new(AtomicBool::new(false)),
        blocking_factory(tx, Arc::clone(&live)),
    );
    let a = scheduler.spawn(Task::read_only("a")).unwrap();
    let b = scheduler.spawn(Task::read_only("b")).unwrap();
    assert_ne!(a, b);
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(live.load(Ordering::SeqCst), 2);
    assert_eq!(
        scheduler.spawn(Task::read_only("over capacity")),
        Err(ScheduleError::Capacity)
    );
    let view = scheduler.view();
    drop(scheduler);
    assert_eq!(live.load(Ordering::SeqCst), 0);
    assert!(
        view.snapshots()
            .iter()
            .all(|s| s.state == TaskState::Finished(Outcome::Cancelled)
                && s.stop_requested == Some(StopRequest::Shutdown))
    );
}

#[test]
fn parent_cancellation_propagates_to_all_children_without_parent_polling() {
    let (tx, rx) = mpsc::sync_channel(4);
    let live = Arc::new(AtomicUsize::new(0));
    let cancel = Arc::new(AtomicBool::new(false));
    let mut scheduler = scheduler(
        Config::default(),
        Arc::clone(&cancel),
        blocking_factory(tx, live),
    );
    let a = scheduler.spawn(Task::read_only("a")).unwrap();
    let b = scheduler.spawn(Task::read_only("b")).unwrap();
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
    cancel.store(true, Ordering::Release);
    for id in [a, b] {
        let snapshot = finished(&scheduler, id);
        assert_eq!(snapshot.stop_requested, Some(StopRequest::ParentCancelled));
        assert_eq!(snapshot.state, TaskState::Finished(Outcome::Cancelled));
    }
    assert_eq!(
        scheduler.spawn(Task::read_only("late")),
        Err(ScheduleError::Closed)
    );
}

#[test]
fn cancelling_one_child_does_not_cancel_its_sibling() {
    let (tx, rx) = mpsc::sync_channel(4);
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        blocking_factory(tx, Arc::new(AtomicUsize::new(0))),
    );
    let a = scheduler.spawn(Task::read_only("a")).unwrap();
    let b = scheduler.spawn(Task::read_only("b")).unwrap();
    for _ in 0..2 {
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
    }
    scheduler.cancel(a).unwrap();
    assert_eq!(
        finished(&scheduler, a).stop_requested,
        Some(StopRequest::ChildCancelled)
    );
    assert_eq!(
        scheduler
            .view()
            .snapshots()
            .into_iter()
            .find(|s| s.id == b)
            .unwrap()
            .state,
        TaskState::Running
    );
}

#[test]
fn ui_control_rejects_stale_ids_cancels_independently_and_closes_future_spawns() {
    let (tx, rx) = mpsc::sync_channel(4);
    let parent_cancel = Arc::new(AtomicBool::new(false));
    let mut scheduler = scheduler(
        Config::default(),
        Arc::clone(&parent_cancel),
        blocking_factory(tx, Arc::new(AtomicUsize::new(0))),
    );
    let a = scheduler.spawn(Task::read_only("a")).unwrap();
    let b = scheduler.spawn(Task::read_only("b")).unwrap();
    for _ in 0..2 {
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
    }
    let control = scheduler.control();
    assert_eq!(
        control.cancel("child-stale-1"),
        Err(ScheduleError::UnknownTask)
    );
    assert!(control.cancel(&a.to_string()).unwrap());
    assert_eq!(
        finished(&scheduler, a).state,
        TaskState::Finished(Outcome::Cancelled)
    );
    assert!(!control.cancel(&a.to_string()).unwrap());
    assert_eq!(
        scheduler.wait(b, Duration::ZERO).unwrap().state,
        TaskState::Running
    );
    assert_eq!(control.cancel_all(), 1);
    assert_eq!(control.cancel_all(), 0);
    assert_eq!(
        scheduler.spawn(Task::read_only("late")),
        Err(ScheduleError::Closed)
    );
    assert_eq!(
        finished(&scheduler, b).state,
        TaskState::Finished(Outcome::Cancelled)
    );
    assert!(!parent_cancel.load(Ordering::Acquire));
    for id in [a, b] {
        assert_eq!(
            scheduler.take_result(id).unwrap().stop_requested,
            Some(StopRequest::ChildCancelled)
        );
    }
}

#[test]
fn child_lifecycle_wakes_registered_ui_without_a_polling_watcher() {
    let (tx, rx) = mpsc::sync_channel(4);
    let mut scheduler = scheduler(
        Config::default(),
        Arc::new(AtomicBool::new(false)),
        blocking_factory(tx, Arc::new(AtomicUsize::new(0))),
    );
    let view = scheduler.view();
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let (woken_tx, woken_rx) = mpsc::sync_channel(1);
    let watcher = thread::spawn(move || {
        view.wake_current_thread();
        let before = view.revision();
        ready_tx.send(()).unwrap();
        while view.revision() == before {
            thread::park_timeout(Duration::from_secs(1));
        }
        woken_tx.send(()).unwrap();
    });
    ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    scheduler.spawn(Task::read_only("wake UI")).unwrap();
    rx.recv_timeout(Duration::from_secs(1)).unwrap();
    woken_rx
        .recv_timeout(Duration::from_millis(500))
        .expect("UI should unpark on lifecycle notification");
    watcher.join().unwrap();
}
