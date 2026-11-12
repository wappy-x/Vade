//! Dedicated CPU prompt packing, independent of HTTP parsing and tokenization.
use std::{cell::RefCell, sync::mpsc};

pub type Task = Box<dyn FnOnce() + Send>;
type Dispatch = Box<dyn Fn(Task) -> Result<(), Task>>;
thread_local! { static DISPATCH: RefCell<Option<Dispatch>> = RefCell::new(None); }

/// Install a packing-only queue for the duration of `run`.
pub fn with_dispatcher<R>(send: impl Fn(Task) -> Result<(), Task> + 'static,
    run: impl FnOnce() -> R) -> R {
    struct Restore(Option<Dispatch>);
    impl Drop for Restore {
        fn drop(&mut self) { DISPATCH.with(|slot| *slot.borrow_mut() = self.0.take()); }
    }
    let _restore = Restore(DISPATCH.with(|slot| slot.replace(Some(Box::new(send)))));
    run()
}

/// A failed dispatch runs locally so owners can always drain their completion.
pub fn submit(task: Task) {
    let remaining = DISPATCH.with(|slot| match slot.borrow().as_ref() {
        Some(send) => send(task).err(), None => Some(task),
    });
    if let Some(task) = remaining { task(); }
}

/// Serving keeps this worker for the scheduler lifetime; native calls reuse it
/// when nested. Drop closes and drains the queue before its owners can return.
pub fn with_worker<R>(run: impl FnOnce() -> R) -> Result<R, String> {
    if DISPATCH.with(|slot| slot.borrow().is_some()) { return Ok(run()); }
    std::thread::scope(|scope| {
        let (send, jobs) = mpsc::sync_channel::<Task>(1);
        std::thread::Builder::new().name("prompt-prep".into()).spawn_scoped(scope, move || {
            #[cfg(target_os = "linux")]
            if let Err(error) = crate::driver::prefer_fast_cores() { eprintln!("prompt preparation: {error}"); }
            while let Ok(task) = jobs.recv() { task(); }
        }).map_err(|e| format!("prompt preparation spawn: {e}"))?;
        Ok(with_dispatcher(move |task| send.send(task).map_err(|e| e.0), run))
    })
}
